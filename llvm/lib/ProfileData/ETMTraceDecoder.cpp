//===-- ETMTraceDecoder.cpp - ETM Trace Decoder -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/ETMTraceDecoder.h"
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

class ETMDecoderImpl : public ETMDecoder {
  dcd_tree_handle_t DcdTree = 0;
  const object::Binary &Binary;
  const Triple &TargetTriple;

  // Trace processing and Callback handling.
  static ocsd_datapath_resp_t
  processTrace(const void *PContext, const ocsd_trc_index_t /*IndexSOP*/,
               const uint8_t /*TrcChanID*/,
               const ocsd_generic_trace_elem *Element) {
    auto *Decoder = static_cast<ETMDecoderImpl *>(const_cast<void *>(PContext));
    if (!Decoder || !Element)
      return OCSD_RESP_FATAL_SYS_ERR;

    // Process instruction ranges reconstructed by the decoder.
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
    return OCSD_RESP_CONT;
  }

  Callback *CurrentCallback = nullptr;

  // Iterate through the ELF program headers to collect all executable LOAD
  // segments. These are registered as a single transaction to the OpenCSD
  // memory manager to prevent overlap/collision errors between different
  // memory regions.
  Error mapELFSegments(dcd_tree_handle_t DcdTree,
                       const object::Binary &SourceBin) {
    SmallVector<ocsd_file_mem_region_t, 4> Regions;
    auto ProcessHeaders = [&](const auto &ElfFile) {
      auto ProgramHeaders = ElfFile.program_headers();
      if (!ProgramHeaders)
        return;

      for (const auto &Phdr : *ProgramHeaders) {
        if (Phdr.p_type == llvm::ELF::PT_LOAD &&
            (Phdr.p_flags & llvm::ELF::PF_X)) {
          ocsd_file_mem_region_t Region{};
          Region.start_address = (uint64_t)Phdr.p_vaddr;
          Region.file_offset = (uint64_t)Phdr.p_offset;
          Region.region_size = (uint64_t)Phdr.p_filesz;
          Regions.push_back(Region);
        }
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

    if (!Regions.empty()) {
      std::string Path = SourceBin.getFileName().str();
      if (ocsd_dt_add_binfile_region_mem_acc(
              DcdTree, Regions.data(), (uint32_t)Regions.size(),
              OCSD_MEM_SPACE_ANY, Path.c_str()) != 0) {
        return createStringError(
            inconvertibleErrorCode(),
            "OpenCSD: Failed to map ELF executable segments.");
      }
    }
    return Error::success();
  }

public:
  uint8_t TraceID;

  ETMDecoderImpl(const object::Binary &Binary, const Triple &Triple,
                 uint8_t TraceID)
      : Binary(Binary), TargetTriple(Triple), TraceID(TraceID) {}

  ~ETMDecoderImpl() override {
    if (DcdTree)
      // Deallocate the decoder tree resources.
      ocsd_destroy_dcd_tree(DcdTree);
  }

  // Initialize the decoder by auto-detecting the target architecture and
  // configuring the OpenCSD decoder.
  Error initialize() {
    DcdTree = ocsd_create_dcd_tree(OCSD_TRC_SRC_SINGLE, 0);
    if (!DcdTree)
      return createStringError(inconvertibleErrorCode(),
                               "Failed to create OpenCSD decoder tree.");

    // Configure and initialize the instruction-level decoder.
    ETMTraceConfig Config(TargetTriple, TraceID);
    if (Error E = Config.validate())
      return E;

    uint32_t Flags =
        OCSD_CREATE_FLG_FULL_DECODER | OCSD_OPFLG_CHK_RANGE_CONTINUE;
    if (ocsd_dt_create_decoder(DcdTree, OCSD_BUILTIN_DCD_ETMV4I, Flags,
                               (void *)&Config.Cfg, &Config.TraceID) != 0)
      return createStringError(
          inconvertibleErrorCode(),
          "OpenCSD: Failed to initialize the instruction decoder.");

    // Extract and map executable segments from the ELF binary.
    if (Error E = mapELFSegments(DcdTree, Binary))
      return E;

    // Register the high-level packet callback. The 'processTrace' function
    // will be invoked for every decoded instruction range.
    ocsd_dt_set_gen_elem_outfn(DcdTree, processTrace, this);
    return Error::success();
  }

  Error processTrace(ArrayRef<uint8_t> TraceData,
                     Callback &TraceCallback) override {
    CurrentCallback = &TraceCallback;
    // Initial reset to prime the decoder.
    ocsd_dt_process_data(DcdTree, OCSD_OP_RESET, 0, 0, nullptr, nullptr);

    const uint8_t *DataPtr = TraceData.data();
    uint32_t TotalSize = TraceData.size();
    uint32_t Processed = 0;

    // Core Decoding Loop.
    while (Processed < TotalSize) {
      uint32_t Consumed = 0;
      uint32_t Remaining = TotalSize - Processed;
      ocsd_datapath_resp_t Response =
          ocsd_dt_process_data(DcdTree, OCSD_OP_DATA, Processed, Remaining,
                               DataPtr + Processed, &Consumed);

      if (Response == OCSD_RESP_WAIT) {
        // Decoder buffers are full; flush to drain internal states.
        ocsd_dt_process_data(DcdTree, OCSD_OP_FLUSH, 0, 0, nullptr, nullptr);
      } else if (Consumed == 0 && Processed < TotalSize) {
        // Decoder stalled; skip byte and reset to find next sync point.
        Processed++;
        ocsd_dt_process_data(DcdTree, OCSD_OP_RESET, 0, 0, nullptr, nullptr);
      } else {
        // Successfully consumed bytes of the bitstream.
        Processed += Consumed;
      }

      if (Response >= OCSD_RESP_FATAL_INVALID_DATA)
        return createStringError(inconvertibleErrorCode(),
                                 "OpenCSD: Fatal decoding error.");
    }

    // Finalize the decoding session by flushing the EOT (End of Trace) marker.
    ocsd_dt_process_data(DcdTree, OCSD_OP_EOT, 0, 0, nullptr, nullptr);
    return Error::success();
  }
};
} // namespace

Expected<std::unique_ptr<ETMDecoder>>
ETMDecoder::create(const object::Binary &Binary, const Triple &Triple,
                   uint8_t TraceID) {
  auto Decoder = std::make_unique<ETMDecoderImpl>(Binary, Triple, TraceID);
  if (Error E = Decoder->initialize())
    return std::move(E);
  return std::unique_ptr<ETMDecoder>(std::move(Decoder));
}

} // namespace llvm

#else // !HAVE_OPENCSD

namespace llvm {

Expected<std::unique_ptr<ETMDecoder>>
ETMDecoder::create(const object::Binary & /*Binary*/, const Triple & /*Triple*/,
                   uint8_t /*TraceID*/) {
  return createStringError(inconvertibleErrorCode(), "OpenCSD not enabled.");
}

} // namespace llvm

#endif // HAVE_OPENCSD
