//===-- Decoder.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Decoder.h"

// C/C++ Includes
#include <cinttypes>
#include <cstring>

#include "lldb/API/SBModule.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBThread.h"

using namespace ptdecoder_private;

// This function removes entries of all the processes/threads which were once
// registered in the class but are not alive anymore because they died or
// finished executing.
void Decoder::RemoveDeadProcessesAndThreads(lldb::SBProcess &sbprocess) {
  lldb::SBTarget sbtarget = sbprocess.GetTarget();
  lldb::SBDebugger sbdebugger = sbtarget.GetDebugger();
  uint32_t num_targets = sbdebugger.GetNumTargets();

  auto itr_process = m_mapProcessUID_mapThreadID_TraceInfo.begin();
  while (itr_process != m_mapProcessUID_mapThreadID_TraceInfo.end()) {
    bool process_found = false;
    lldb::SBTarget target;
    lldb::SBProcess process;
    for (uint32_t i = 0; i < num_targets; i++) {
      target = sbdebugger.GetTargetAtIndex(i);
      process = target.GetProcess();
      if (process.GetUniqueID() == itr_process->first) {
        process_found = true;
        break;
      }
    }

    // Remove the process's entry if it was not found in SBDebugger
    if (!process_found) {
      itr_process = m_mapProcessUID_mapThreadID_TraceInfo.erase(itr_process);
      continue;
    }

    // If the state of the process is exited or detached then remove process's
    // entry. If not then remove entry for all those registered threads of this
    // process that are not alive anymore.
    lldb::StateType state = process.GetState();
    if ((state == lldb::StateType::eStateDetached) ||
        (state == lldb::StateType::eStateExited))
      itr_process = m_mapProcessUID_mapThreadID_TraceInfo.erase(itr_process);
    else {
      auto itr_thread = itr_process->second.begin();
      while (itr_thread != itr_process->second.end()) {
        if (itr_thread->first == LLDB_INVALID_THREAD_ID) {
          ++itr_thread;
          continue;
        }

        lldb::SBThread thread = process.GetThreadByID(itr_thread->first);
        if (!thread.IsValid())
          itr_thread = itr_process->second.erase(itr_thread);
        else
          ++itr_thread;
      }
      ++itr_process;
    }
  }
}

void Decoder::StartProcessorTrace(lldb::SBProcess &sbprocess,
                                  lldb::SBTraceOptions &sbtraceoptions,
                                  lldb::SBError &sberror) {
  sberror.Clear();
  CheckDebuggerID(sbprocess, sberror);
  if (!sberror.Success())
    return;

  std::lock_guard<std::mutex> guard(
      m_mapProcessUID_mapThreadID_TraceInfo_mutex);
  RemoveDeadProcessesAndThreads(sbprocess);

  if (sbtraceoptions.getType() != lldb::TraceType::eTraceTypeProcessorTrace) {
    sberror.SetErrorStringWithFormat("SBTraceOptions::TraceType not set to "
                                     "eTraceTypeProcessorTrace; ProcessID = "
                                     "%" PRIu64,
                                     sbprocess.GetProcessID());
    return;
  }
  lldb::SBStructuredData sbstructdata = sbtraceoptions.getTraceParams(sberror);
  if (!sberror.Success())
    return;

  const char *trace_tech_key = "trace-tech";
  std::string trace_tech_value("intel-pt");
  lldb::SBStructuredData value = sbstructdata.GetValueForKey(trace_tech_key);
  if (!value.IsValid()) {
    sberror.SetErrorStringWithFormat(
        "key \"%s\" not set in custom trace parameters", trace_tech_key);
    return;
  }

  char string_value[9];
  size_t bytes_written = value.GetStringValue(
      string_value, sizeof(string_value) / sizeof(*string_value));
  if (!bytes_written ||
      (bytes_written > (sizeof(string_value) / sizeof(*string_value)))) {
    sberror.SetErrorStringWithFormat(
        "key \"%s\" not set in custom trace parameters", trace_tech_key);
    return;
  }

  std::size_t pos =
      trace_tech_value.find((const char *)string_value, 0, bytes_written);
  if ((pos == std::string::npos)) {
    sberror.SetErrorStringWithFormat(
        "key \"%s\" not set to \"%s\" in custom trace parameters",
        trace_tech_key, trace_tech_value.c_str());
    return;
  }

  // Start Tracing
  lldb::SBError error;
  uint32_t unique_id = sbprocess.GetUniqueID();
  lldb::tid_t tid = sbtraceoptions.getThreadID();
  lldb::SBTrace trace = sbprocess.StartTrace(sbtraceoptions, error);
  if (!error.Success()) {
    if (tid == LLDB_INVALID_THREAD_ID)
      sberror.SetErrorStringWithFormat("%s; ProcessID = %" PRIu64,
                                       error.GetCString(),
                                       sbprocess.GetProcessID());
    else
      sberror.SetErrorStringWithFormat(
          "%s; thread_id = %" PRIu64 ", ProcessID = %" PRIu64,
          error.GetCString(), tid, sbprocess.GetProcessID());
    return;
  }

  MapThreadID_TraceInfo &mapThreadID_TraceInfo =
      m_mapProcessUID_mapThreadID_TraceInfo[unique_id];
  ThreadTraceInfo &trace_info = mapThreadID_TraceInfo[tid];
  trace_info.SetUniqueTraceInstance(trace);
  trace_info.SetStopID(sbprocess.GetStopID());
}

void Decoder::StopProcessorTrace(lldb::SBProcess &sbprocess,
                                 lldb::SBError &sberror, lldb::tid_t tid) {
  sberror.Clear();
  CheckDebuggerID(sbprocess, sberror);
  if (!sberror.Success()) {
    return;
  }

  std::lock_guard<std::mutex> guard(
      m_mapProcessUID_mapThreadID_TraceInfo_mutex);
  RemoveDeadProcessesAndThreads(sbprocess);

  uint32_t unique_id = sbprocess.GetUniqueID();
  auto itr_process = m_mapProcessUID_mapThreadID_TraceInfo.find(unique_id);
  if (itr_process == m_mapProcessUID_mapThreadID_TraceInfo.end()) {
    sberror.SetErrorStringWithFormat(
        "tracing not active for this process; ProcessID = %" PRIu64,
        sbprocess.GetProcessID());
    return;
  }

  lldb::SBError error;
  if (tid == LLDB_INVALID_THREAD_ID) {
    // This implies to stop tracing on the whole process
    lldb::user_id_t id_to_be_ignored = LLDB_INVALID_UID;
    auto itr_thread = itr_process->second.begin();
    while (itr_thread != itr_process->second.end()) {
      // In the case when user started trace on the entire process and then
      // registered newly spawned threads of this process in the class later,
      // these newly spawned threads will have same trace id. If we stopped
      // trace on the entire process then tracing stops automatically for these
      // newly spawned registered threads. Stopping trace on them again will
      // return error and therefore we need to skip stopping trace on them
      // again.
      lldb::SBTrace &trace = itr_thread->second.GetUniqueTraceInstance();
      lldb::user_id_t lldb_pt_user_id = trace.GetTraceUID();
      if (lldb_pt_user_id != id_to_be_ignored) {
        trace.StopTrace(error, itr_thread->first);
        if (!error.Success()) {
          std::string error_string(error.GetCString());
          if ((error_string.find("tracing not active for this process") ==
               std::string::npos) &&
              (error_string.find("tracing not active for this thread") ==
               std::string::npos)) {
            sberror.SetErrorStringWithFormat(
                "%s; thread id=%" PRIu64 ", ProcessID = %" PRIu64,
                error_string.c_str(), itr_thread->first,
                sbprocess.GetProcessID());
            return;
          }
        }

        if (itr_thread->first == LLDB_INVALID_THREAD_ID)
          id_to_be_ignored = lldb_pt_user_id;
      }
      itr_thread = itr_process->second.erase(itr_thread);
    }
    m_mapProcessUID_mapThreadID_TraceInfo.erase(itr_process);
  } else {
    // This implies to stop tracing on a single thread.
    // if 'tid' is registered in the class then get the trace id and stop trace
    // on it. If it is not then check if tracing was ever started on the entire
    // process (because there is a possibility that trace is still running for
    // 'tid' but it was not registered in the class because user had started
    // trace on the whole process and 'tid' spawned later). In that case, get
    // the trace id of the process trace instance and stop trace on this thread.
    // If tracing was never started on the entire process then return error
    // because there is no way tracing is active on 'tid'.
    MapThreadID_TraceInfo &mapThreadID_TraceInfo = itr_process->second;
    lldb::SBTrace trace;
    auto itr = mapThreadID_TraceInfo.find(tid);
    if (itr != mapThreadID_TraceInfo.end()) {
      trace = itr->second.GetUniqueTraceInstance();
    } else {
      auto itr = mapThreadID_TraceInfo.find(LLDB_INVALID_THREAD_ID);
      if (itr != mapThreadID_TraceInfo.end()) {
        trace = itr->second.GetUniqueTraceInstance();
      } else {
        sberror.SetErrorStringWithFormat(
            "tracing not active for this thread; thread id=%" PRIu64
            ", ProcessID = %" PRIu64,
            tid, sbprocess.GetProcessID());
        return;
      }
    }

    // Stop Tracing
    trace.StopTrace(error, tid);
    if (!error.Success()) {
      std::string error_string(error.GetCString());
      sberror.SetErrorStringWithFormat(
          "%s; thread id=%" PRIu64 ", ProcessID = %" PRIu64,
          error_string.c_str(), tid, sbprocess.GetProcessID());
      if (error_string.find("tracing not active") == std::string::npos)
        return;
    }
    // Delete the entry of 'tid' from this class (if any)
    mapThreadID_TraceInfo.erase(tid);
  }
}

void Decoder::ReadTraceDataAndImageInfo(lldb::SBProcess &sbprocess,
                                        lldb::tid_t tid, lldb::SBError &sberror,
                                        ThreadTraceInfo &threadTraceInfo) {
  // Allocate trace data buffer and parse cpu info for 'tid' if it is registered
  // for the first time in class
  lldb::SBTrace &trace = threadTraceInfo.GetUniqueTraceInstance();
  Buffer &pt_buffer = threadTraceInfo.GetPTBuffer();
  lldb::SBError error;
  if (pt_buffer.size() == 0) {
    lldb::SBTraceOptions traceoptions;
    traceoptions.setThreadID(tid);
    trace.GetTraceConfig(traceoptions, error);
    if (!error.Success()) {
      sberror.SetErrorStringWithFormat("%s; ProcessID = %" PRIu64,
                                       error.GetCString(),
                                       sbprocess.GetProcessID());
      return;
    }
    if (traceoptions.getType() != lldb::TraceType::eTraceTypeProcessorTrace) {
      sberror.SetErrorStringWithFormat("invalid TraceType received from LLDB "
                                       "for this thread; thread id=%" PRIu64
                                       ", ProcessID = %" PRIu64,
                                       tid, sbprocess.GetProcessID());
      return;
    }

    threadTraceInfo.AllocatePTBuffer(traceoptions.getTraceBufferSize());
    lldb::SBStructuredData sbstructdata = traceoptions.getTraceParams(sberror);
    if (!sberror.Success())
      return;
    CPUInfo &pt_cpu = threadTraceInfo.GetCPUInfo();
    ParseCPUInfo(pt_cpu, sbstructdata, sberror);
    if (!sberror.Success())
      return;
  }

  // Call LLDB API to get raw trace data for this thread
  size_t bytes_written = trace.GetTraceData(error, (void *)pt_buffer.data(),
                                            pt_buffer.size(), 0, tid);
  if (!error.Success()) {
    sberror.SetErrorStringWithFormat(
        "%s; thread_id = %" PRIu64 ",  ProcessID = %" PRIu64,
        error.GetCString(), tid, sbprocess.GetProcessID());
    return;
  }
  std::fill(pt_buffer.begin() + bytes_written, pt_buffer.end(), 0);

  // Get information of all the modules of the inferior
  lldb::SBTarget sbtarget = sbprocess.GetTarget();
  ReadExecuteSectionInfos &readExecuteSectionInfos =
      threadTraceInfo.GetReadExecuteSectionInfos();
  GetTargetModulesInfo(sbtarget, readExecuteSectionInfos, sberror);
  if (!sberror.Success())
    return;
}

void Decoder::DecodeProcessorTrace(lldb::SBProcess &sbprocess, lldb::tid_t tid,
                                   lldb::SBError &sberror,
                                   ThreadTraceInfo &threadTraceInfo) {
  // Initialize instruction decoder
  struct pt_insn_decoder *decoder = nullptr;
  struct pt_config config;
  Buffer &pt_buffer = threadTraceInfo.GetPTBuffer();
  CPUInfo &pt_cpu = threadTraceInfo.GetCPUInfo();
  ReadExecuteSectionInfos &readExecuteSectionInfos =
      threadTraceInfo.GetReadExecuteSectionInfos();

  InitializePTInstDecoder(&decoder, &config, pt_cpu, pt_buffer,
                          readExecuteSectionInfos, sberror);
  if (!sberror.Success())
    return;

  // Start raw trace decoding
  Instructions &instruction_list = threadTraceInfo.GetInstructionLog();
  instruction_list.clear();
  DecodeTrace(decoder, instruction_list, sberror);
}

// Raw trace decoding requires information of Read & Execute sections of each
// module of the inferior. This function updates internal state of the class to
// store this information.
void Decoder::GetTargetModulesInfo(
    lldb::SBTarget &sbtarget, ReadExecuteSectionInfos &readExecuteSectionInfos,
    lldb::SBError &sberror) {
  if (!sbtarget.IsValid()) {
    sberror.SetErrorStringWithFormat("Can't get target's modules info from "
                                     "LLDB; process has an invalid target");
    return;
  }

  lldb::SBFileSpec target_file_spec = sbtarget.GetExecutable();
  if (!target_file_spec.IsValid()) {
    sberror.SetErrorStringWithFormat("Target has an invalid file spec");
    return;
  }

  uint32_t num_modules = sbtarget.GetNumModules();
  readExecuteSectionInfos.clear();

  // Store information of all RX sections of each module of inferior
  for (uint32_t i = 0; i < num_modules; i++) {
    lldb::SBModule module = sbtarget.GetModuleAtIndex(i);
    if (!module.IsValid()) {
      sberror.SetErrorStringWithFormat(
          "Can't get module info [ %" PRIu32
          " ] of target \"%s\" from LLDB, invalid module",
          i, target_file_spec.GetFilename());
      return;
    }

    lldb::SBFileSpec module_file_spec = module.GetPlatformFileSpec();
    if (!module_file_spec.IsValid()) {
      sberror.SetErrorStringWithFormat(
          "Can't get module info [ %" PRIu32
          " ] of target \"%s\" from LLDB, invalid file spec",
          i, target_file_spec.GetFilename());
      return;
    }

    const char *image(module_file_spec.GetFilename());
    lldb::SBError error;
    char image_complete_path[1024];
    uint32_t path_length = module_file_spec.GetPath(
        image_complete_path, sizeof(image_complete_path));
    size_t num_sections = module.GetNumSections();

    // Store information of only RX sections
    for (size_t idx = 0; idx < num_sections; idx++) {
      lldb::SBSection section = module.GetSectionAtIndex(idx);
      uint32_t section_permission = section.GetPermissions();
      if ((section_permission & lldb::Permissions::ePermissionsReadable) &&
          (section_permission & lldb::Permissions::ePermissionsExecutable)) {
        lldb::SBData section_data = section.GetSectionData();
        if (!section_data.IsValid()) {
          sberror.SetErrorStringWithFormat(
              "Can't get module info [ %" PRIu32 " ]   \"%s\"  of target "
              "\"%s\" from LLDB, invalid "
              "data in \"%s\" section",
              i, image, target_file_spec.GetFilename(), section.GetName());
          return;
        }

        // In case section has no data, skip it.
        if (section_data.GetByteSize() == 0)
          continue;

        if (!path_length) {
          sberror.SetErrorStringWithFormat(
              "Can't get module info [ %" PRIu32 " ]   \"%s\"  of target "
              "\"%s\" from LLDB, module "
              "has an invalid path length",
              i, image, target_file_spec.GetFilename());
          return;
        }

        std::string image_path(image_complete_path, path_length);
        readExecuteSectionInfos.emplace_back(
            section.GetLoadAddress(sbtarget), section.GetFileOffset(),
            section_data.GetByteSize(), image_path);
      }
    }
  }
}

// Raw trace decoding requires information of the target cpu on which inferior
// is running. This function gets the Trace Configuration from LLDB, parses it
// for cpu model, family, stepping and vendor id info and updates the internal
// state of the class to store this information.
void Decoder::ParseCPUInfo(CPUInfo &pt_cpu, lldb::SBStructuredData &s,
                           lldb::SBError &sberror) {
  lldb::SBStructuredData custom_trace_params = s.GetValueForKey("intel-pt");
  if (!custom_trace_params.IsValid()) {
    sberror.SetErrorStringWithFormat("lldb couldn't provide cpuinfo");
    return;
  }

  uint64_t family = 0, model = 0, stepping = 0;
  char vendor[32];
  const char *key_family = "cpu_family";
  const char *key_model = "cpu_model";
  const char *key_stepping = "cpu_stepping";
  const char *key_vendor = "cpu_vendor";

  // parse family
  lldb::SBStructuredData struct_family =
      custom_trace_params.GetValueForKey(key_family);
  if (!struct_family.IsValid()) {
    sberror.SetErrorStringWithFormat(
        "%s info missing in custom trace parameters", key_family);
    return;
  }
  family = struct_family.GetIntegerValue(0x10000);
  if (family > UINT16_MAX) {
    sberror.SetErrorStringWithFormat(
        "invalid CPU family value extracted from custom trace parameters");
    return;
  }
  pt_cpu.family = (uint16_t)family;

  // parse model
  lldb::SBStructuredData struct_model =
      custom_trace_params.GetValueForKey(key_model);
  if (!struct_model.IsValid()) {
    sberror.SetErrorStringWithFormat(
        "%s info missing in custom trace parameters; family=%" PRIu16,
        key_model, pt_cpu.family);
    return;
  }
  model = struct_model.GetIntegerValue(0x100);
  if (model > UINT8_MAX) {
    sberror.SetErrorStringWithFormat("invalid CPU model value extracted from "
                                     "custom trace parameters; family=%" PRIu16,
                                     pt_cpu.family);
    return;
  }
  pt_cpu.model = (uint8_t)model;

  // parse stepping
  lldb::SBStructuredData struct_stepping =
      custom_trace_params.GetValueForKey(key_stepping);
  if (!struct_stepping.IsValid()) {
    sberror.SetErrorStringWithFormat(
        "%s info missing in custom trace parameters; family=%" PRIu16
        ", model=%" PRIu8,
        key_stepping, pt_cpu.family, pt_cpu.model);
    return;
  }
  stepping = struct_stepping.GetIntegerValue(0x100);
  if (stepping > UINT8_MAX) {
    sberror.SetErrorStringWithFormat("invalid CPU stepping value extracted "
                                     "from custom trace parameters; "
                                     "family=%" PRIu16 ", model=%" PRIu8,
                                     pt_cpu.family, pt_cpu.model);
    return;
  }
  pt_cpu.stepping = (uint8_t)stepping;

  // parse vendor info
  pt_cpu.vendor = pcv_unknown;
  lldb::SBStructuredData struct_vendor =
      custom_trace_params.GetValueForKey(key_vendor);
  if (!struct_vendor.IsValid()) {
    sberror.SetErrorStringWithFormat(
        "%s info missing in custom trace parameters; family=%" PRIu16
        ", model=%" PRIu8 ", stepping=%" PRIu8,
        key_vendor, pt_cpu.family, pt_cpu.model, pt_cpu.stepping);
    return;
  }
  auto length = struct_vendor.GetStringValue(vendor, sizeof(vendor));
  if (length && strstr(vendor, "GenuineIntel"))
    pt_cpu.vendor = pcv_intel;
}

// Initialize trace decoder with pt_config structure and populate its image
// structure with inferior's memory image information. pt_config structure is
// initialized with trace buffer and cpu info of the inferior before storing it
// in trace decoder.
void Decoder::InitializePTInstDecoder(
    struct pt_insn_decoder **decoder, struct pt_config *config,
    const CPUInfo &pt_cpu, Buffer &pt_buffer,
    const ReadExecuteSectionInfos &readExecuteSectionInfos,
    lldb::SBError &sberror) const {
  if (!decoder || !config) {
    sberror.SetErrorStringWithFormat("internal error");
    return;
  }

  // Load cpu info of inferior's target in pt_config struct
  pt_config_init(config);
  config->cpu = pt_cpu;
  int errcode = pt_cpu_errata(&(config->errata), &(config->cpu));
  if (errcode < 0) {
    sberror.SetErrorStringWithFormat("processor trace decoding library: "
                                     "pt_cpu_errata() failed with error: "
                                     "\"%s\"",
                                     pt_errstr(pt_errcode(errcode)));
    return;
  }

  // Load trace buffer's starting and end address in pt_config struct
  config->begin = pt_buffer.data();
  config->end = pt_buffer.data() + pt_buffer.size();

  // Fill trace decoder with pt_config struct
  *decoder = pt_insn_alloc_decoder(config);
  if (*decoder == nullptr) {
    sberror.SetErrorStringWithFormat("processor trace decoding library:  "
                                     "pt_insn_alloc_decoder() returned null "
                                     "pointer");
    return;
  }

  // Fill trace decoder's image with inferior's memory image information
  struct pt_image *image = pt_insn_get_image(*decoder);
  if (!image) {
    sberror.SetErrorStringWithFormat("processor trace decoding library:  "
                                     "pt_insn_get_image() returned null "
                                     "pointer");
    pt_insn_free_decoder(*decoder);
    return;
  }

  for (auto &itr : readExecuteSectionInfos) {
    errcode = pt_image_add_file(image, itr.image_path.c_str(), itr.file_offset,
                                itr.size, nullptr, itr.load_address);
    if (errcode < 0) {
      sberror.SetErrorStringWithFormat("processor trace decoding library:  "
                                       "pt_image_add_file() failed with error: "
                                       "\"%s\"",
                                       pt_errstr(pt_errcode(errcode)));
      pt_insn_free_decoder(*decoder);
      return;
    }
  }
}

// Start actual decoding of raw trace
void Decoder::DecodeTrace(struct pt_insn_decoder *decoder,
                          Instructions &instruction_list,
                          lldb::SBError &sberror) {
  uint64_t decoder_offset = 0;

  while (1) {
    struct pt_insn insn;

    // Try to sync the decoder. If it fails then get the decoder_offset and try
    // to sync again. If the new_decoder_offset is same as decoder_offset then
    // we will not succeed in syncing for any number of pt_insn_sync_forward()
    // operations. Return in that case. Else keep resyncing until either end of
    // trace stream is reached or pt_insn_sync_forward() passes.
    int errcode = pt_insn_sync_forward(decoder);
    if (errcode < 0) {
      if (errcode == -pte_eos)
        return;

      int errcode_off = pt_insn_get_offset(decoder, &decoder_offset);
      if (errcode_off < 0) {
        sberror.SetErrorStringWithFormat(
            "processor trace decoding library: \"%s\"",
            pt_errstr(pt_errcode(errcode)));
        instruction_list.emplace_back(sberror.GetCString());
        return;
      }

      sberror.SetErrorStringWithFormat(
          "processor trace decoding library: \"%s\"  [decoder_offset] => "
          "[0x%" PRIu64 "]",
          pt_errstr(pt_errcode(errcode)), decoder_offset);
      instruction_list.emplace_back(sberror.GetCString());
      while (1) {
        errcode = pt_insn_sync_forward(decoder);
        if (errcode >= 0)
          break;

        if (errcode == -pte_eos)
          return;

        uint64_t new_decoder_offset = 0;
        errcode_off = pt_insn_get_offset(decoder, &new_decoder_offset);
        if (errcode_off < 0) {
          sberror.SetErrorStringWithFormat(
              "processor trace decoding library: \"%s\"",
              pt_errstr(pt_errcode(errcode)));
          instruction_list.emplace_back(sberror.GetCString());
          return;
        } else if (new_decoder_offset <= decoder_offset) {
          // We tried resyncing the decoder and decoder didn't make any
          // progress because the offset didn't change. We will not make any
          // progress further. Hence, returning in this situation.
          return;
        }
        sberror.SetErrorStringWithFormat(
            "processor trace decoding library: \"%s\"  [decoder_offset] => "
            "[0x%" PRIu64 "]",
            pt_errstr(pt_errcode(errcode)), new_decoder_offset);
        instruction_list.emplace_back(sberror.GetCString());
        decoder_offset = new_decoder_offset;
      }
    }

    while (1) {
      errcode = pt_insn_next(decoder, &insn, sizeof(insn));
      if (errcode < 0) {
        if (insn.iclass == ptic_error)
          break;

        instruction_list.emplace_back(insn);

        if (errcode == -pte_eos)
          return;

        Diagnose(decoder, errcode, sberror, &insn);
        instruction_list.emplace_back(sberror.GetCString());
        break;
      }
      instruction_list.emplace_back(insn);
      if (errcode & pts_eos)
        return;
    }
  }
}

// Function to diagnose and indicate errors during raw trace decoding
void Decoder::Diagnose(struct pt_insn_decoder *decoder, int decode_error,
                       lldb::SBError &sberror, const struct pt_insn *insn) {
  int errcode;
  uint64_t offset;

  errcode = pt_insn_get_offset(decoder, &offset);
  if (insn) {
    if (errcode < 0)
      sberror.SetErrorStringWithFormat(
          "processor trace decoding library: \"%s\"  [decoder_offset, "
          "last_successful_decoded_ip] => [?, 0x%" PRIu64 "]",
          pt_errstr(pt_errcode(decode_error)), insn->ip);
    else
      sberror.SetErrorStringWithFormat(
          "processor trace decoding library: \"%s\"  [decoder_offset, "
          "last_successful_decoded_ip] => [0x%" PRIu64 ", 0x%" PRIu64 "]",
          pt_errstr(pt_errcode(decode_error)), offset, insn->ip);
  } else {
    if (errcode < 0)
      sberror.SetErrorStringWithFormat(
          "processor trace decoding library: \"%s\"",
          pt_errstr(pt_errcode(decode_error)));
    else
      sberror.SetErrorStringWithFormat(
          "processor trace decoding library: \"%s\"  [decoder_offset] => "
          "[0x%" PRIu64 "]",
          pt_errstr(pt_errcode(decode_error)), offset);
  }
}

void Decoder::GetInstructionLogAtOffset(lldb::SBProcess &sbprocess,
                                        lldb::tid_t tid, uint32_t offset,
                                        uint32_t count,
                                        InstructionList &result_list,
                                        lldb::SBError &sberror) {
  sberror.Clear();
  CheckDebuggerID(sbprocess, sberror);
  if (!sberror.Success()) {
    return;
  }

  std::lock_guard<std::mutex> guard(
      m_mapProcessUID_mapThreadID_TraceInfo_mutex);
  RemoveDeadProcessesAndThreads(sbprocess);

  ThreadTraceInfo *threadTraceInfo = nullptr;
  FetchAndDecode(sbprocess, tid, sberror, &threadTraceInfo);
  if (!sberror.Success()) {
    return;
  }
  if (threadTraceInfo == nullptr) {
    sberror.SetErrorStringWithFormat("internal error");
    return;
  }

  // Return instruction log by populating 'result_list'
  Instructions &insn_list = threadTraceInfo->GetInstructionLog();
  uint64_t sum = (uint64_t)offset + 1;
  if (((insn_list.size() <= offset) && (count <= sum) &&
       ((sum - count) >= insn_list.size())) ||
      (count < 1)) {
    sberror.SetErrorStringWithFormat(
        "Instruction Log not available for offset=%" PRIu32
        " and count=%" PRIu32 ", ProcessID = %" PRIu64,
        offset, count, sbprocess.GetProcessID());
    return;
  }

  Instructions::iterator itr_first =
      (insn_list.size() <= offset) ? insn_list.begin()
                                   : insn_list.begin() + insn_list.size() - sum;
  Instructions::iterator itr_last =
      (count <= sum) ? insn_list.begin() + insn_list.size() - (sum - count)
                     : insn_list.end();
  Instructions::iterator itr = itr_first;
  while (itr != itr_last) {
    result_list.AppendInstruction(*itr);
    ++itr;
  }
}

void Decoder::GetProcessorTraceInfo(lldb::SBProcess &sbprocess, lldb::tid_t tid,
                                    TraceOptions &options,
                                    lldb::SBError &sberror) {
  sberror.Clear();
  CheckDebuggerID(sbprocess, sberror);
  if (!sberror.Success()) {
    return;
  }

  std::lock_guard<std::mutex> guard(
      m_mapProcessUID_mapThreadID_TraceInfo_mutex);
  RemoveDeadProcessesAndThreads(sbprocess);

  ThreadTraceInfo *threadTraceInfo = nullptr;
  FetchAndDecode(sbprocess, tid, sberror, &threadTraceInfo);
  if (!sberror.Success()) {
    return;
  }
  if (threadTraceInfo == nullptr) {
    sberror.SetErrorStringWithFormat("internal error");
    return;
  }

  // Get SBTraceOptions from LLDB for 'tid', populate 'traceoptions' with it
  lldb::SBTrace &trace = threadTraceInfo->GetUniqueTraceInstance();
  lldb::SBTraceOptions traceoptions;
  lldb::SBError error;
  traceoptions.setThreadID(tid);
  trace.GetTraceConfig(traceoptions, error);
  if (!error.Success()) {
    std::string error_string(error.GetCString());
    if (error_string.find("tracing not active") != std::string::npos) {
      uint32_t unique_id = sbprocess.GetUniqueID();
      auto itr_process = m_mapProcessUID_mapThreadID_TraceInfo.find(unique_id);
      if (itr_process == m_mapProcessUID_mapThreadID_TraceInfo.end())
        return;
      itr_process->second.erase(tid);
    }
    sberror.SetErrorStringWithFormat("%s; ProcessID = %" PRIu64,
                                     error_string.c_str(),
                                     sbprocess.GetProcessID());
    return;
  }
  if (traceoptions.getType() != lldb::TraceType::eTraceTypeProcessorTrace) {
    sberror.SetErrorStringWithFormat("invalid TraceType received from LLDB "
                                     "for this thread; thread id=%" PRIu64
                                     ", ProcessID = %" PRIu64,
                                     tid, sbprocess.GetProcessID());
    return;
  }
  options.setType(traceoptions.getType());
  options.setTraceBufferSize(traceoptions.getTraceBufferSize());
  options.setMetaDataBufferSize(traceoptions.getMetaDataBufferSize());
  lldb::SBStructuredData sbstructdata = traceoptions.getTraceParams(sberror);
  if (!sberror.Success())
    return;
  options.setTraceParams(sbstructdata);
  options.setInstructionLogSize(threadTraceInfo->GetInstructionLog().size());
}

void Decoder::FetchAndDecode(lldb::SBProcess &sbprocess, lldb::tid_t tid,
                             lldb::SBError &sberror,
                             ThreadTraceInfo **threadTraceInfo) {
  // Return with error if 'sbprocess' is not registered in the class
  uint32_t unique_id = sbprocess.GetUniqueID();
  auto itr_process = m_mapProcessUID_mapThreadID_TraceInfo.find(unique_id);
  if (itr_process == m_mapProcessUID_mapThreadID_TraceInfo.end()) {
    sberror.SetErrorStringWithFormat(
        "tracing not active for this process; ProcessID = %" PRIu64,
        sbprocess.GetProcessID());
    return;
  }

  if (tid == LLDB_INVALID_THREAD_ID) {
    sberror.SetErrorStringWithFormat(
        "invalid thread id provided; thread_id = %" PRIu64
        ", ProcessID = %" PRIu64,
        tid, sbprocess.GetProcessID());
    return;
  }

  // Check whether 'tid' thread is registered in the class. If it is then in
  // case StopID didn't change then return without doing anything (no need to
  // read and decode trace data then). Otherwise, save new StopID and proceed
  // with reading and decoding trace.
  if (threadTraceInfo == nullptr) {
    sberror.SetErrorStringWithFormat("internal error");
    return;
  }

  MapThreadID_TraceInfo &mapThreadID_TraceInfo = itr_process->second;
  auto itr_thread = mapThreadID_TraceInfo.find(tid);
  if (itr_thread != mapThreadID_TraceInfo.end()) {
    if (itr_thread->second.GetStopID() == sbprocess.GetStopID()) {
      *threadTraceInfo = &(itr_thread->second);
      return;
    }
    itr_thread->second.SetStopID(sbprocess.GetStopID());
  } else {
    // Implies 'tid' is not registered in the class. If tracing was never
    // started on the entire process then return an error. Else try to register
    // this thread and proceed with reading and decoding trace.
    lldb::SBError error;
    itr_thread = mapThreadID_TraceInfo.find(LLDB_INVALID_THREAD_ID);
    if (itr_thread == mapThreadID_TraceInfo.end()) {
      sberror.SetErrorStringWithFormat(
          "tracing not active for this thread; ProcessID = %" PRIu64,
          sbprocess.GetProcessID());
      return;
    }

    lldb::SBTrace &trace = itr_thread->second.GetUniqueTraceInstance();
    ThreadTraceInfo &trace_info = mapThreadID_TraceInfo[tid];
    trace_info.SetUniqueTraceInstance(trace);
    trace_info.SetStopID(sbprocess.GetStopID());
    itr_thread = mapThreadID_TraceInfo.find(tid);
  }

  // Get raw trace data and inferior image from LLDB for the registered thread
  ReadTraceDataAndImageInfo(sbprocess, tid, sberror, itr_thread->second);
  if (!sberror.Success()) {
    std::string error_string(sberror.GetCString());
    if (error_string.find("tracing not active") != std::string::npos)
      mapThreadID_TraceInfo.erase(itr_thread);
    return;
  }
  // Decode raw trace data
  DecodeProcessorTrace(sbprocess, tid, sberror, itr_thread->second);
  if (!sberror.Success()) {
    return;
  }
  *threadTraceInfo = &(itr_thread->second);
}

// This function checks whether the provided SBProcess instance belongs to same
// SBDebugger with which this tool instance is associated.
void Decoder::CheckDebuggerID(lldb::SBProcess &sbprocess,
                              lldb::SBError &sberror) {
  if (!sbprocess.IsValid()) {
    sberror.SetErrorStringWithFormat("invalid process instance");
    return;
  }

  lldb::SBTarget sbtarget = sbprocess.GetTarget();
  if (!sbtarget.IsValid()) {
    sberror.SetErrorStringWithFormat(
        "process contains an invalid target; ProcessID = %" PRIu64,
        sbprocess.GetProcessID());
    return;
  }

  lldb::SBDebugger sbdebugger = sbtarget.GetDebugger();
  if (!sbdebugger.IsValid()) {
    sberror.SetErrorStringWithFormat("process's target contains an invalid "
                                     "debugger instance; ProcessID = %" PRIu64,
                                     sbprocess.GetProcessID());
    return;
  }

  if (sbdebugger.GetID() != m_debugger_user_id) {
    sberror.SetErrorStringWithFormat(
        "process belongs to a different SBDebugger instance than the one for "
        "which the tool is instantiated; ProcessID = %" PRIu64,
        sbprocess.GetProcessID());
    return;
  }
}
