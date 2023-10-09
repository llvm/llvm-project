#include "../include/InternalEvent.h"

#include <iomanip>
#include <sstream>

using namespace omptest;

/// String manipulation helper function. Takes up to 8 bytes of data and returns
/// their hexadecimal representation as string. The data can be truncated to a
/// certain size in bytes and will by default be prefixed with '0x'.
std::string makeHexString(uint64_t Data, bool IsPointer = true,
                          size_t DataBytes = 0, bool ShowHexBase = true) {
  if (Data == 0 && IsPointer)
    return "(nil)";

  static std::ostringstream os;
  // Clear the content of the stream
  os.str(std::string());

  // Manually prefixing "0x" will make the use of std::setfill more easy
  if (ShowHexBase)
    os << "0x";

  // Default to 32bit (8 hex digits) width if exceeding 64bit or zero value
  size_t NumDigits = (DataBytes > 0 && DataBytes < 9) ? (DataBytes << 1) : 8;

  if (DataBytes > 0)
    os << std::setfill('0') << std::setw(NumDigits);

  os << std::hex << Data;
  return os.str();
}

std::string internal::ThreadBegin::toString() const {
  std::string S{"OMPT Callback ThreadBegin: "};
  S.append("ThreadType=").append(std::to_string(ThreadType));
  return S;
}

std::string internal::ParallelBegin::toString() const {
  std::string S{"OMPT Callback ParallelBegin: "};
  S.append("NumThreads=").append(std::to_string(NumThreads));
  return S;
}

std::string internal::ParallelEnd::toString() const {
  std::string S{"OMPT Callback ParallelEnd"};
  return S;
}

std::string internal::Target::toString() const {
  std::string S{"Callback Target: target_id="};
  S.append(std::to_string(TargetId));
  S.append(" kind=").append(std::to_string(Kind));
  S.append(" endpoint=").append(std::to_string(Endpoint));
  S.append(" device_num=").append(std::to_string(DeviceNum));
  S.append(" code=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string internal::TargetEmi::toString() const {
  std::string S{"Callback Target EMI: kind="};
  S.append(std::to_string(Kind));
  S.append(" endpoint=").append(std::to_string(Endpoint));
  S.append(" device_num=").append(std::to_string(DeviceNum));
  S.append(" task_data=").append(makeHexString((uint64_t)TaskData));
  S.append(" (")
      .append(makeHexString((uint64_t)TaskData->value, /*IsPointer=*/false))
      .append(1, ')');
  S.append(" target_task_data=")
      .append(makeHexString((uint64_t)TargetTaskData));
  S.append(" (")
      .append(
          makeHexString((uint64_t)TargetTaskData->value, /*IsPointer=*/false))
      .append(1, ')');
  S.append(" target_data=").append(makeHexString((uint64_t)TargetData));
  S.append(" (")
      .append(makeHexString((uint64_t)TargetData->value, /*IsPointer=*/false))
      .append(1, ')');
  S.append(" code=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string internal::TargetDataOp::toString() const {
  std::string S{"  Callback DataOp: target_id="};
  S.append(std::to_string(TargetId));
  S.append(" host_op_id=").append(std::to_string(HostOpId));
  S.append(" optype=").append(std::to_string(OpType));
  S.append(" src=").append(makeHexString((uint64_t)SrcAddr));
  S.append(" src_device_num=").append(std::to_string(SrcDeviceNum));
  S.append(" dest=").append(makeHexString((uint64_t)DstAddr));
  S.append(" dest_device_num=").append(std::to_string(DstDeviceNum));
  S.append(" bytes=").append(std::to_string(Bytes));
  S.append(" code=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string internal::TargetDataOpEmi::toString() const {
  std::string S{"  Callback DataOp EMI: endpoint="};
  S.append(std::to_string(Endpoint));
  S.append(" optype=").append(std::to_string(OpType));
  S.append(" target_task_data=")
      .append(makeHexString((uint64_t)TargetTaskData));
  S.append(" (")
      .append(
          makeHexString((uint64_t)TargetTaskData->value, /*IsPointer=*/false))
      .append(1, ')');
  S.append(" target_data=").append(makeHexString((uint64_t)TargetData));
  S.append(" (")
      .append(makeHexString((uint64_t)TargetData->value, /*IsPointer=*/false))
      .append(1, ')');
  S.append(" host_op_id=").append(makeHexString((uint64_t)HostOpId));
  S.append(" (")
      .append(makeHexString((uint64_t)(*HostOpId), /*IsPointer=*/false))
      .append(1, ')');
  S.append(" src=").append(makeHexString((uint64_t)SrcAddr));
  S.append(" src_device_num=").append(std::to_string(SrcDeviceNum));
  S.append(" dest=").append(makeHexString((uint64_t)DstAddr));
  S.append(" dest_device_num=").append(std::to_string(DstDeviceNum));
  S.append(" bytes=").append(std::to_string(Bytes));
  S.append(" code=").append(makeHexString((uint64_t)CodeptrRA));
  return S;
}

std::string internal::TargetSubmit::toString() const {
  std::string S{"  Callback Submit: target_id="};
  S.append(std::to_string(TargetId));
  S.append(" host_op_id=").append(std::to_string(HostOpId));
  S.append(" req_num_teams=").append(std::to_string(RequestedNumTeams));
  return S;
}

std::string internal::TargetSubmitEmi::toString() const {
  std::string S{"  Callback Submit EMI: endpoint="};
  S.append(std::to_string(Endpoint));
  S.append("  req_num_teams=").append(std::to_string(RequestedNumTeams));
  S.append(" target_data=").append(makeHexString((uint64_t)TargetData));
  S.append(" (")
      .append(makeHexString((uint64_t)TargetData->value, /*IsPointer=*/false))
      .append(1, ')');
  S.append(" host_op_id=").append(makeHexString((uint64_t)HostOpId));
  S.append(" (")
      .append(makeHexString((uint64_t)(*HostOpId), /*IsPointer=*/false))
      .append(1, ')');
  return S;
}

std::string internal::DeviceInitialize::toString() const {
  std::string S{"Callback Init: device_num="};
  S.append(std::to_string(DeviceNum));
  S.append(" type=").append(Type);
  S.append(" device=").append(makeHexString((uint64_t)Device));
  S.append(" lookup=").append(makeHexString((uint64_t)LookupFn));
  S.append(" doc=").append(makeHexString((uint64_t)DocStr));
  return S;
}

std::string internal::DeviceFinalize::toString() const {
  std::string S{"Callback Fini: device_num="};
  S.append(std::to_string(DeviceNum));
  return S;
}

std::string internal::DeviceLoad::toString() const {
  std::string S{"Callback Load: device_num:"};
  S.append(std::to_string(DeviceNum));
  S.append(" module_id:").append(std::to_string(ModuleId));
  S.append(" filename:").append((Filename == nullptr) ? "(null)" : Filename);
  S.append(" host_adddr:").append(makeHexString((uint64_t)HostAddr));
  S.append(" device_addr:").append(makeHexString((uint64_t)DeviceAddr));
  S.append(" bytes:").append(std::to_string(Bytes));
  return S;
}

std::string internal::BufferRequest::toString() const {
  std::string S{"Allocated "};
  S.append(std::to_string(*Bytes)).append(" bytes at ");
  S.append(makeHexString((uint64_t)*Buffer));
  S.append(" in buffer request callback");
  return S;
}

std::string internal::BufferComplete::toString() const {
  std::string S{"Executing buffer complete callback: "};
  S.append(std::to_string(DeviceNum)).append(1, ' ');
  S.append(makeHexString((uint64_t)Buffer)).append(1, ' ');
  S.append(std::to_string(Bytes)).append(1, ' ');
  S.append(makeHexString((uint64_t)Begin)).append(1, ' ');
  S.append(std::to_string(BufferOwned));
  return S;
}

std::string internal::BufferRecord::toString() const {
  std::string S{""};
  // First line
  S.append("rec=").append(makeHexString((uint64_t)RecordPtr));
  S.append(" type=").append(std::to_string(Record.type));
  S.append(" time=").append(std::to_string(Record.time));
  S.append(" thread_id=").append(std::to_string(Record.thread_id));
  S.append(" target_id=").append(std::to_string(Record.target_id));
  S.append(1, '\n');

  // Second line
  switch (Record.type) {
  case ompt_callback_target:
  case ompt_callback_target_emi: {
    // Handle Target Record
    ompt_record_target_t TR = Record.record.target;
    printf("\tRecord Target: kind=%d endpoint=%d device=%d task_id=%lu "
           "target_id=%lu codeptr=%p\n",
           TR.kind, TR.endpoint, TR.device_num, TR.task_id, TR.target_id,
           TR.codeptr_ra);
    S.append("\tRecord Target: kind=").append(std::to_string(TR.kind));
    S.append(" endpoint=").append(std::to_string(TR.endpoint));
    S.append(" device=").append(std::to_string(TR.device_num));
    S.append(" task_id=").append(std::to_string(TR.task_id));
    S.append(" target_id=").append(std::to_string(TR.target_id));
    S.append(" codeptr=").append(makeHexString((uint64_t)TR.codeptr_ra));
    break;
  }
  case ompt_callback_target_data_op:
  case ompt_callback_target_data_op_emi: {
    // Handle Target DataOp Record
    ompt_record_target_data_op_t TDR = Record.record.target_data_op;
    S.append("\t  Record DataOp: host_op_id=")
        .append(std::to_string(TDR.host_op_id));
    S.append(" optype=").append(std::to_string(TDR.optype));
    S.append(" src_addr=").append(makeHexString((uint64_t)TDR.src_addr));
    S.append(" src_device=").append(std::to_string(TDR.src_device_num));
    S.append(" dest_addr=").append(makeHexString((uint64_t)TDR.dest_addr));
    S.append(" dest_device=").append(std::to_string(TDR.dest_device_num));
    S.append(" bytes=").append(std::to_string(TDR.bytes));
    S.append(" end_time=").append(std::to_string(TDR.end_time));
    S.append(" duration=").append(std::to_string(TDR.end_time - Record.time));
    S.append(" ns codeptr=").append(makeHexString((uint64_t)TDR.codeptr_ra));
    break;
  }
  case ompt_callback_target_submit:
  case ompt_callback_target_submit_emi: {
    // Handle Target Kernel Record
    ompt_record_target_kernel_t TKR = Record.record.target_kernel;
    S.append("\t  Record Submit: host_op_id=")
        .append(std::to_string(TKR.host_op_id));
    S.append(" requested_num_teams=")
        .append(std::to_string(TKR.requested_num_teams));
    S.append(" granted_num_teams=")
        .append(std::to_string(TKR.granted_num_teams));
    S.append(" end_time=").append(std::to_string(TKR.end_time));
    S.append(" duration=").append(std::to_string(TKR.end_time - Record.time));
    S.append(" ns");
    break;
  }
  default:
    S.append("Unsupported record type: ").append(std::to_string(Record.type));
    break;
  }

  return S;
}
