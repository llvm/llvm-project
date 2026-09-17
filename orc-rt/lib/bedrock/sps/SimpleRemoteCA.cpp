//===- SimpleRemoteCA.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transport-independent half of the SimpleRemote protocol.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/sps/SimpleRemoteCA.h"

#include "orc-rt/support/Compiler.h"
#include "orc-rt/support/iterator_range.h"
#include "orc-rt/support/sps/SimplePackedSerialization.h"

#include <string>
#include <string_view>
#include <utility>

namespace orc_rt {

const char *SimpleRemoteCA::getOpcodeName(Opcode Op) noexcept {
  switch (Op) {
  case Opcode::Setup:
    return "setup";
  case Opcode::Hangup:
    return "hang-up";
  case Opcode::Result:
    return "result";
  case Opcode::Call:
    return "call";
  }
  ORC_RT_UNREACHABLE("Unrecognized opcode");
}

WrapperFunctionBuffer SimpleRemoteCA::encodeSetup(const BootstrapInfo &BI) {
  using SPSSetup = SPSTuple<SPSString, uint64_t,
                            SPSSequence<SPSTuple<SPSString, SPSSequence<char>>>,
                            SPSSequence<SPSTuple<SPSString, SPSExecutorAddr>>>;

  // Force uint64_t PageSize.
  // FIXME: Remove once we allow size_t serialization.
  uint64_t PageSize = BI.processInfo().pageSize();
  auto Symbols = iterator_range(BI.symbols());
  auto Tuple =
      std::tie(BI.processInfo().targetTriple(), PageSize, BI.values(), Symbols);
  using Serialize = SPSSerializationTraits<SPSSetup, decltype(Tuple)>;

  auto Payload = WrapperFunctionBuffer::allocate(Serialize::size(Tuple));
  SPSOutputBuffer OB(Payload.data(), Payload.size());
  if (!Serialize::serialize(OB, Tuple))
    ORC_RT_UNREACHABLE("serialization should not fail");
  return Payload;
}

WrapperFunctionBuffer SimpleRemoteCA::encodeHangup(Error Err) {
  SPSSerializableError SE(std::move(Err));
  using Serialize = SPSArgList<SPSError>;
  auto Payload = WrapperFunctionBuffer::allocate(Serialize::size(SE));
  SPSOutputBuffer OB(Payload.data(), Payload.size());
  if (!Serialize::serialize(OB, SE))
    ORC_RT_UNREACHABLE("serialization should not fail");
  return Payload;
}

Error SimpleRemoteCA::decodeHangup(WrapperFunctionBuffer Payload) {
  SPSSerializableError SE;
  SPSInputBuffer IB(Payload.data(), Payload.size());
  if (!SPSArgList<SPSError>::deserialize(IB, SE))
    return make_error<StringError>(
        "Malformed hang-up message: could not deserialize reason");
  return SE.toError();
}

std::pair<SimpleRemoteCA::ResultKind, WrapperFunctionBuffer>
SimpleRemoteCA::encodeResult(WrapperFunctionBuffer ResultBytes) {
  const char *ErrMsg = ResultBytes.getOutOfBandError();
  if (!ErrMsg)
    return {ResultKind::Value, std::move(ResultBytes)};

  using Serialize = SPSArgList<SPSString>;
  std::string_view Msg(ErrMsg);
  auto Payload = WrapperFunctionBuffer::allocate(Serialize::size(Msg));
  SPSOutputBuffer OB(Payload.data(), Payload.size());
  if (!Serialize::serialize(OB, Msg))
    ORC_RT_UNREACHABLE("serialization should not fail");
  return {ResultKind::OutOfBandError, std::move(Payload)};
}

WrapperFunctionBuffer
SimpleRemoteCA::decodeResult(ResultKind Kind, WrapperFunctionBuffer Payload) {
  switch (Kind) {
  case ResultKind::Value:
    return Payload;
  case ResultKind::OutOfBandError: {
    // A payload that will not decode becomes the out-of-band error itself: the
    // call waiting on this result has to be completed either way, and an error
    // about the error tells the caller more than a dead session would.
    std::string Msg;
    SPSInputBuffer IB(Payload.data(), Payload.size());
    if (!SPSArgList<SPSString>::deserialize(IB, Msg))
      return WrapperFunctionBuffer::createOutOfBandError(
          "Malformed result message: could not deserialize out-of-band error");
    return WrapperFunctionBuffer::createOutOfBandError(Msg.c_str());
  }
  }
  ORC_RT_UNREACHABLE("Unrecognized result kind");
}

uint64_t SimpleRemoteCA::registerCall(OnControllerCallReturn OnComplete) {
  assert(OnComplete && "Registered handler must contain a value");
  uint64_t SeqNo = NextSeqNo++;
  PendingCalls.try_emplace(SeqNo, std::move(OnComplete));
  return SeqNo;
}

SimpleRemoteCA::OnControllerCallReturn
SimpleRemoteCA::takeCall(uint64_t SeqNo) {
  auto I = PendingCalls.find(SeqNo);
  if (I == PendingCalls.end())
    return OnControllerCallReturn();
  auto OnComplete = std::move(I->second);
  PendingCalls.erase(I);
  return OnComplete;
}

SimpleRemoteCA::PendingCallsMap SimpleRemoteCA::takeAllCalls() {
  PendingCallsMap Taken;
  std::swap(Taken, PendingCalls);
  return Taken;
}

Expected<SimpleRemoteCA::Action>
SimpleRemoteCA::handleMessage(uint64_t OpC, uint64_t SeqNo, ExecutorAddr Tag,
                              WrapperFunctionBuffer Payload) {
  if (OpC > static_cast<uint64_t>(Opcode::LastOpcode))
    return make_error<StringError>("Invalid opcode " + std::to_string(OpC));

  switch (static_cast<Opcode>(OpC)) {
  case Opcode::Setup:
    return make_error<StringError>("Unexpected Setup message");

  case Opcode::Hangup: {
    // A hang-up carries no sequence number or tag, and a payload holding the
    // reason the controller is going away.
    if (SeqNo != 0 || Tag)
      return make_error<StringError>("Malformed hang-up message");
    // A reason ends the session with that reason; an orderly hang-up, or a
    // payload that will not decode, just ends it.
    if (Error Err = decodeHangup(std::move(Payload)))
      return std::move(Err);
    return Action::End;
  }

  case Opcode::Result: {
    // The tag carries the result kind rather than a handler tag. Checked here,
    // so that decodeResult only ever sees a kind it can interpret.
    uint64_t KindVal = Tag.getValue();
    if (KindVal > static_cast<uint64_t>(ResultKind::LastResultKind))
      return make_error<StringError>("Malformed result message: invalid kind " +
                                     std::to_string(KindVal));
    if (auto Err = handleResult(SeqNo, static_cast<ResultKind>(KindVal),
                                std::move(Payload)))
      return std::move(Err);
    return Action::Continue;
  }

  case Opcode::Call:
    handleWrapperCall(Tag.toPtr<orc_rt_WrapperFunction>(), std::move(Payload),
                      SeqNo);
    return Action::Continue;
  }
  ORC_RT_UNREACHABLE("Unrecognized opcode");
}

Error SimpleRemoteCA::handleResult(uint64_t SeqNo, ResultKind Kind,
                                   WrapperFunctionBuffer ResultBytes) {
  auto OnComplete = takePendingCall(SeqNo);
  if (!OnComplete)
    return make_error<StringError>("No pending call for sequence number " +
                                   std::to_string(SeqNo));

  handleControllerCallResult(std::move(OnComplete),
                             decodeResult(Kind, std::move(ResultBytes)));
  return Error::success();
}

} // namespace orc_rt
