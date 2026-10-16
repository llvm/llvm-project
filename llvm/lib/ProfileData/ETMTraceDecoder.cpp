//===-- ETMTraceDecoder.cpp - ETM Trace Decoder -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/ETMTraceDecoder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/ARMTargetParser.h"

#ifdef HAVE_OPENCSD
#include "opencsd/c_api/opencsd_c_api.h"

namespace llvm {

namespace {

class HardwareTraceConfig {
public:
  virtual ~HardwareTraceConfig() = default;
};

class ETMTraceConfig : public HardwareTraceConfig {
public:
  ocsd_etmv4_cfg Cfg{};
  uint8_t TraceID;

  ETMTraceConfig(const Triple &TargetTriple, uint8_t TraceID)
      : TraceID(TraceID) {
    ocsd_arch_version_t ArchVer = ARCH_UNKNOWN;
    if (TargetTriple.isArmMClass()) {
      unsigned ArchVersion = ARM::parseArchVersion(TargetTriple.getArchName());
      if (ArchVersion >= 8)
        ArchVer = ARCH_V8;
      else if (ArchVersion == 7)
        ArchVer = ARCH_V7;
      else
        // For version 6 (Cortex-M0) and others.
        ArchVer = ARCH_UNKNOWN;
    }
    // Initialize the decoder for Arm M-profile targets.
    Cfg.arch_ver = ArchVer;
    Cfg.core_prof = profile_CortexM;

    // The CoreSight Trace ID (CSID) is a hardware-assigned 7-bit identifier
    // used to route trace data.
    Cfg.reg_traceidr = TraceID;
  }

  Error validate() const {
    if (Cfg.arch_ver == ARCH_UNKNOWN)
      return createStringError(
          inconvertibleErrorCode(),
          "OpenCSD: Unsupported processor architecture. Only Arm M-profile "
          "(Cortex-M) with ETM support is currently supported.");
    return Error::success();
  }
};

class ITMTraceConfig : public HardwareTraceConfig {
public:
  ocsd_itm_cfg Cfg{};
  uint8_t TraceID;

  ITMTraceConfig(uint8_t TraceID) : TraceID(TraceID) {
    // The 7-bit CoreSight Trace ID is stored in bits [22:16] of the ITM Trace
    // Control Register (ITM_TCR).
    Cfg.reg_tcr = static_cast<uint32_t>(TraceID) << 16;
  }
};

class ETMDecoderImpl : public ETMDecoder {
  dcd_tree_handle_t ETMDcdTree = 0;
  dcd_tree_handle_t ITMDcdTree = 0;
  dcd_tree_handle_t MultiplexedDcdTree = 0;
  const object::Binary &Binary;
  const Triple &TargetTriple;
  SmallVector<ocsd_file_mem_region_t> CodeRegions;
  SmallVector<ocsd_file_mem_region_t> DataRegions;

  uint64_t resolveDataAddress(const swt_itm_info &ITM) const {
    // ITM data address packets carry a 1-, 2-, or 4-byte address payload.
    uint64_t Address = ITM.value;
    if (ITM.payload_size >= 4)
      return Address;

    // For 1- or 2-byte packets, the hardware omits the upper address bytes;
    // fill in the missing upper bits from the ELF binary's data/BSS sections.
    uint64_t UpperBitsMask = ~((1ULL << (ITM.payload_size * 8)) - 1);
    for (const auto &Region : DataRegions) {
      uint64_t Start = Region.start_address;
      uint64_t End = Start + Region.region_size;
      uint64_t FullAddress = (Start & UpperBitsMask) | Address;
      if (FullAddress >= Start && FullAddress < End)
        return FullAddress;
    }

    return Address;
  }

  // Trace processing and Callback handling.
  static ocsd_datapath_resp_t
  processTrace(const void *PContext, const ocsd_trc_index_t /*IndexSOP*/,
               const uint8_t /*TrcChanID*/,
               const ocsd_generic_trace_elem *Element) {
    auto *Decoder = static_cast<ETMDecoderImpl *>(const_cast<void *>(PContext));
    if (!Decoder || !Element)
      return OCSD_RESP_FATAL_SYS_ERR;

    // Process instruction ranges reconstructed from the ETM trace.
    if (Element->elem_type == OCSD_GEN_TRC_ELEM_INSTR_RANGE) {
      uint64_t Start = Element->st_addr;
      uint64_t End = Element->en_addr;
      if (End > Start) {
        // OpenCSD ranges are exclusive at the end [Start, End).
        // llvm-profgen range counters expect inclusive bounds [Start, End].
        // Adjust the exclusive end address provided by OpenCSD to include
        // the last executed instruction within the reported range.
        Decoder->CurrentCallback->processInstructionRange(Start, End - 1);
      }
    }

    // Process data addresses from the ITM trace.
    if (Element->elem_type == OCSD_GEN_TRC_ELEM_ITMTRACE &&
        Element->swt_itm.pkt_type == DWT_PAYLOAD) {
      uint64_t Address = Decoder->resolveDataAddress(Element->swt_itm);
      Decoder->CurrentCallback->processDataAddress(Address);
    }

    return OCSD_RESP_CONT;
  }

  Callback *CurrentCallback = nullptr;

  // Iterate through the ELF program headers to collect executable code regions.
  void collectCodeRegions(const object::Binary &SourceBin) {
    auto ProcessHeaders = [&](const auto &ElfFile) {
      auto ProgramHeaders = ElfFile.program_headers();
      if (!ProgramHeaders)
        return;

      for (const auto &Phdr : *ProgramHeaders) {
        if (Phdr.p_type != llvm::ELF::PT_LOAD ||
            !(Phdr.p_flags & llvm::ELF::PF_X))
          continue;

        ocsd_file_mem_region_t Region{};
        Region.start_address = (uint64_t)Phdr.p_vaddr;
        Region.file_offset = (uint64_t)Phdr.p_offset;
        Region.region_size = (uint64_t)Phdr.p_filesz;
        CodeRegions.push_back(Region);
      }
    };

    if (auto *O = dyn_cast<object::ELF32LEObjectFile>(&SourceBin))
      ProcessHeaders(O->getELFFile());
    else if (auto *O = dyn_cast<object::ELF64LEObjectFile>(&SourceBin))
      ProcessHeaders(O->getELFFile());
    else if (auto *O = dyn_cast<object::ELF32BEObjectFile>(&SourceBin))
      ProcessHeaders(O->getELFFile());
    else if (auto *O = dyn_cast<object::ELF64BEObjectFile>(&SourceBin))
      ProcessHeaders(O->getELFFile());
  }

  // Iterate through the ELF sections to collect data and BSS regions.
  void collectDataRegions(const object::Binary &SourceBin) {
    const auto *Obj = dyn_cast<object::ObjectFile>(&SourceBin);
    if (!Obj)
      return;

    for (const object::SectionRef &Sec : Obj->sections()) {
      if ((!Sec.isData() && !Sec.isBSS()) || Sec.getSize() == 0)
        continue;

      ocsd_file_mem_region_t Region{};
      Region.start_address = Sec.getAddress();
      Region.region_size = Sec.getSize();
      DataRegions.push_back(Region);
    }
  }

  Error createETMDecoder(dcd_tree_handle_t DcdTree) {
    // Configure and initialize the instruction-level decoder.
    ETMTraceConfig ETMConfig(TargetTriple, ETMTraceID);
    if (Error E = ETMConfig.validate())
      return E;

    uint32_t ETMFlags =
        OCSD_CREATE_FLG_FULL_DECODER | OCSD_OPFLG_CHK_RANGE_CONTINUE;
    if (ocsd_dt_create_decoder(DcdTree, OCSD_BUILTIN_DCD_ETMV4I, ETMFlags,
                               (void *)&ETMConfig.Cfg, &ETMConfig.TraceID) != 0)
      return createStringError(
          inconvertibleErrorCode(),
          "OpenCSD: Failed to initialize the instruction decoder.");

    if (CodeRegions.empty())
      return Error::success();

    // Map executable code regions as a single transaction to the OpenCSD
    // memory manager to prevent overlap/collision errors between different
    // memory regions.
    std::string Path = Binary.getFileName().str();
    if (ocsd_dt_add_binfile_region_mem_acc(
            DcdTree, CodeRegions.data(), (uint32_t)CodeRegions.size(),
            OCSD_MEM_SPACE_ANY, Path.c_str()) != 0) {
      return createStringError(
          inconvertibleErrorCode(),
          "OpenCSD: Failed to map ELF executable segments.");
    }

    return Error::success();
  }

  Error createITMDecoder(dcd_tree_handle_t DcdTree) {
    // Configure and initialize the ITM data decoder.
    ITMTraceConfig ITMConfig(ITMTraceID);
    uint32_t ITMFlags =
        OCSD_CREATE_FLG_FULL_DECODER | OCSD_OPFLG_PKTPROC_UNSYNC_ON_BAD_PKTS;
    if (ocsd_dt_create_decoder(DcdTree, OCSD_BUILTIN_DCD_ITM, ITMFlags,
                               (void *)&ITMConfig.Cfg, &ITMConfig.TraceID) != 0)
      return createStringError(
          inconvertibleErrorCode(),
          "OpenCSD: Failed to initialize the ITM data decoder.");

    return Error::success();
  }

  Error createDcdTree(dcd_tree_handle_t &DcdTree, bool EnableETM,
                      bool EnableITM) {
    if (EnableETM && EnableITM)
      DcdTree = ocsd_create_dcd_tree(OCSD_TRC_SRC_FRAME_FORMATTED,
                                     OCSD_DFRMTR_HAS_FSYNCS);
    else
      DcdTree = ocsd_create_dcd_tree(OCSD_TRC_SRC_SINGLE, 0);

    if (!DcdTree)
      return createStringError(inconvertibleErrorCode(),
                               "Failed to create OpenCSD decoder tree.");

    if (EnableETM)
      if (Error E = createETMDecoder(DcdTree))
        return E;

    if (EnableITM)
      if (Error E = createITMDecoder(DcdTree))
        return E;

    ocsd_dt_set_gen_elem_outfn(DcdTree, processTrace, this);
    return Error::success();
  }

  Error decodeDcdTree(dcd_tree_handle_t Tree, ArrayRef<uint8_t> Data) {
    // Initial reset to prime the decoder.
    ocsd_dt_process_data(Tree, OCSD_OP_RESET, 0, 0, nullptr, nullptr);

    const uint8_t *DataPtr = Data.data();
    uint32_t TotalSize = Data.size();
    uint32_t Processed = 0;

    // Core Decoding Loop.
    while (Processed < TotalSize) {
      uint32_t Consumed = 0;
      uint32_t Remaining = TotalSize - Processed;
      ocsd_datapath_resp_t Response =
          ocsd_dt_process_data(Tree, OCSD_OP_DATA, Processed, Remaining,
                               DataPtr + Processed, &Consumed);

      if (Response == OCSD_RESP_WAIT) {
        // Decoder buffers are full; flush to drain internal states.
        ocsd_dt_process_data(Tree, OCSD_OP_FLUSH, 0, 0, nullptr, nullptr);
      } else if (Consumed == 0 && Processed < TotalSize) {
        // Decoder stalled; skip byte and reset to find next sync point.
        Processed++;
        ocsd_dt_process_data(Tree, OCSD_OP_RESET, 0, 0, nullptr, nullptr);
      } else {
        // Successfully consumed bytes of the bitstream.
        Processed += Consumed;
      }

      if (Response >= OCSD_RESP_FATAL_INVALID_DATA)
        return createStringError(inconvertibleErrorCode(),
                                 "OpenCSD: Fatal decoding error.");
    }

    // Finalize the decoding session by flushing the EOT (End of Trace) marker.
    ocsd_dt_process_data(Tree, OCSD_OP_EOT, 0, 0, nullptr, nullptr);
    return Error::success();
  }

public:
  uint8_t ETMTraceID;
  uint8_t ITMTraceID;

  ETMDecoderImpl(const object::Binary &Binary, const Triple &Triple,
                 uint8_t ETMTraceID, uint8_t ITMTraceID)
      : Binary(Binary), TargetTriple(Triple), ETMTraceID(ETMTraceID),
        ITMTraceID(ITMTraceID) {}

  ~ETMDecoderImpl() override {
    if (ETMDcdTree)
      // Deallocate the decoder tree resources.
      ocsd_destroy_dcd_tree(ETMDcdTree);
    if (ITMDcdTree)
      ocsd_destroy_dcd_tree(ITMDcdTree);
    if (MultiplexedDcdTree)
      ocsd_destroy_dcd_tree(MultiplexedDcdTree);
  }

  // Initialize the decoder by auto-detecting the target architecture and
  // configuring the OpenCSD decoders.
  Error initialize() {
    collectCodeRegions(Binary);
    collectDataRegions(Binary);

    if (Error E =
            createDcdTree(ETMDcdTree, /*EnableETM=*/true, /*EnableITM=*/false))
      return E;
    if (Error E =
            createDcdTree(ITMDcdTree, /*EnableETM=*/false, /*EnableITM=*/true))
      return E;
    if (Error E = createDcdTree(MultiplexedDcdTree, /*EnableETM=*/true,
                                /*EnableITM=*/true))
      return E;

    return Error::success();
  }

  // 4-byte CoreSight frame synchronization packet (0x7FFFFFFF in little-endian
  // byte order).
  static constexpr uint8_t FSyncPacket[] = {0xFF, 0xFF, 0xFF, 0x7F};

  Error processTrace(ArrayRef<uint8_t> TraceData,
                     Callback &TraceCallback) override {
    CurrentCallback = &TraceCallback;

    // A trace starting with a CoreSight frame synchronization packet is a
    // multiplexed trace containing both ETM and ITM traces.
    if (TraceData.take_front(sizeof(FSyncPacket)) == ArrayRef(FSyncPacket))
      return decodeDcdTree(MultiplexedDcdTree, TraceData);

    // Search for the initial synchronization packet to discard any leading
    // truncated packets and select the matching decoder.
    size_t NumZeros = 0;
    for (size_t I = 0; I < TraceData.size(); ++I) {
      // Count consecutive zero bytes preceding the 0x80 synchronization byte.
      if (TraceData[I] == 0x00) {
        ++NumZeros;
        continue;
      }

      // ETM synchronization packet: >= 11 zero bytes followed by 0x80.
      // (00 00 00 00 00 00 00 00 00 00 00 80)
      if (TraceData[I] == 0x80 && NumZeros >= 11)
        return decodeDcdTree(ETMDcdTree, TraceData.slice(I - NumZeros));

      // ITM synchronization packet: >= 5 zero bytes followed by 0x80.
      // (00 00 00 00 00 80)
      if (TraceData[I] == 0x80 && NumZeros >= 5)
        return decodeDcdTree(ITMDcdTree, TraceData.slice(I - NumZeros));

      NumZeros = 0;
    }

    return createStringError(
        inconvertibleErrorCode(),
        "No synchronization header (0x80) found in the bitstream.");
  }
};
} // namespace

Expected<std::unique_ptr<ETMDecoder>>
ETMDecoder::create(const object::Binary &Binary, const Triple &Triple,
                   uint8_t ETMTraceID, uint8_t ITMTraceID) {
  auto Decoder =
      std::make_unique<ETMDecoderImpl>(Binary, Triple, ETMTraceID, ITMTraceID);
  if (Error E = Decoder->initialize())
    return std::move(E);
  return std::unique_ptr<ETMDecoder>(std::move(Decoder));
}

} // namespace llvm

#else // !HAVE_OPENCSD

namespace llvm {

Expected<std::unique_ptr<ETMDecoder>>
ETMDecoder::create(const object::Binary & /*Binary*/, const Triple & /*Triple*/,
                   uint8_t /*ETMTraceID*/, uint8_t /*ITMTraceID*/) {
  return createStringError(inconvertibleErrorCode(), "OpenCSD not enabled.");
}

} // namespace llvm

#endif // HAVE_OPENCSD
