//===-- X86PfmCounters.td - X86 Hardware Counters ----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This describes the available hardware counters for various subtargets.
//
//===----------------------------------------------------------------------===//

def UnhaltedCoreCyclesPfmCounter : PfmCounter<"unhalted_core_cycles">;
def UopsIssuedPfmCounter : PfmCounter<"uops_issued:any">;

// No default counters on X86.
def DefaultPfmCounters : ProcPfmCounters {}
def : PfmCountersDefaultBinding<DefaultPfmCounters>;

def SandyBridgePfmCounters : ProcPfmCounters {
  let CycleCounter = UnhaltedCoreCyclesPfmCounter;
  let UopsCounter = UopsIssuedPfmCounter;
  let IssueCounters = [
    PfmIssueCounter<"SBPort0",  "uops_dispatched_port:port_0">,
    PfmIssueCounter<"SBPort1",  "uops_dispatched_port:port_1">,
    PfmIssueCounter<"SBPort23", "uops_dispatched_port:port_2 + uops_dispatched_port:port_3">,
    PfmIssueCounter<"SBPort4",  "uops_dispatched_port:port_4">,
    PfmIssueCounter<"SBPort5",  "uops_dispatched_port:port_5">
  ];
}
def : PfmCountersBinding<"sandybridge", SandyBridgePfmCounters>;

def HaswellPfmCounters : ProcPfmCounters {
  let CycleCounter = UnhaltedCoreCyclesPfmCounter;
  let UopsCounter = UopsIssuedPfmCounter;
  let IssueCounters = [
    PfmIssueCounter<"HWPort0", "uops_dispatched_port:port_0">,
    PfmIssueCounter<"HWPort1", "uops_dispatched_port:port_1">,
    PfmIssueCounter<"HWPort2", "uops_dispatched_port:port_2">,
    PfmIssueCounter<"HWPort3", "uops_dispatched_port:port_3">,
    PfmIssueCounter<"HWPort4", "uops_dispatched_port:port_4">,
    PfmIssueCounter<"HWPort5", "uops_dispatched_port:port_5">,
    PfmIssueCounter<"HWPort6", "uops_dispatched_port:port_6">,
    PfmIssueCounter<"HWPort7", "uops_dispatched_port:port_7">
  ];
}
def : PfmCountersBinding<"haswell", HaswellPfmCounters>;

def BroadwellPfmCounters : ProcPfmCounters {
  let CycleCounter = UnhaltedCoreCyclesPfmCounter;
  let UopsCounter = UopsIssuedPfmCounter;
  let IssueCounters = [
    PfmIssueCounter<"BWPort0", "uops_executed_port:port_0">,
    PfmIssueCounter<"BWPort1", "uops_executed_port:port_1">,
    PfmIssueCounter<"BWPort2", "uops_executed_port:port_2">,
    PfmIssueCounter<"BWPort3", "uops_executed_port:port_3">,
    PfmIssueCounter<"BWPort4", "uops_executed_port:port_4">,
    PfmIssueCounter<"BWPort5", "uops_executed_port:port_5">,
    PfmIssueCounter<"BWPort6", "uops_executed_port:port_6">,
    PfmIssueCounter<"BWPort7", "uops_executed_port:port_7">
  ];
}
def : PfmCountersBinding<"broadwell", BroadwellPfmCounters>;

def SkylakeClientPfmCounters : ProcPfmCounters {
  let CycleCounter = UnhaltedCoreCyclesPfmCounter;
  let UopsCounter = UopsIssuedPfmCounter;
  let IssueCounters = [
    PfmIssueCounter<"SKLPort0", "uops_dispatched_port:port_0">,
    PfmIssueCounter<"SKLPort1", "uops_dispatched_port:port_1">,
    PfmIssueCounter<"SKLPort2", "uops_dispatched_port:port_2">,
    PfmIssueCounter<"SKLPort3", "uops_dispatched_port:port_3">,
    PfmIssueCounter<"SKLPort4", "uops_dispatched_port:port_4">,
    PfmIssueCounter<"SKLPort5", "uops_dispatched_port:port_5">,
    PfmIssueCounter<"SKLPort6", "uops_dispatched_port:port_6">,
    PfmIssueCounter<"SKLPort7", "uops_dispatched_port:port_7">
  ];
}
def : PfmCountersBinding<"skylake", SkylakeClientPfmCounters>;

def SkylakeServerPfmCounters : ProcPfmCounters {
  let CycleCounter = UnhaltedCoreCyclesPfmCounter;
  let UopsCounter = UopsIssuedPfmCounter;
  let IssueCounters = [
    PfmIssueCounter<"SKXPort0", "uops_dispatched_port:port_0">,
    PfmIssueCounter<"SKXPort1", "uops_dispatched_port:port_1">,
    PfmIssueCounter<"SKXPort2", "uops_dispatched_port:port_2">,
    PfmIssueCounter<"SKXPort3", "uops_dispatched_port:port_3">,
    PfmIssueCounter<"SKXPort4", "uops_dispatched_port:port_4">,
    PfmIssueCounter<"SKXPort5", "uops_dispatched_port:port_5">,
    PfmIssueCounter<"SKXPort6", "uops_dispatched_port:port_6">,
    PfmIssueCounter<"SKXPort7", "uops_dispatched_port:port_7">
  ];
}
def : PfmCountersBinding<"skylake-avx512", SkylakeServerPfmCounters>;

def BdVer2PfmCounters : ProcPfmCounters {
  let CycleCounter = PfmCounter<"cpu_clk_unhalted">;
  let UopsCounter = PfmCounter<"retired_uops">;
  let IssueCounters = [
    PfmIssueCounter<"PdFPU0", "dispatched_fpu_ops:ops_pipe0 + dispatched_fpu_ops:ops_dual_pipe0">,
    PfmIssueCounter<"PdFPU1", "dispatched_fpu_ops:ops_pipe1 + dispatched_fpu_ops:ops_dual_pipe1">,
    PfmIssueCounter<"PdFPU2", "dispatched_fpu_ops:ops_pipe2 + dispatched_fpu_ops:ops_dual_pipe2">,
    PfmIssueCounter<"PdFPU3", "dispatched_fpu_ops:ops_pipe3 + dispatched_fpu_ops:ops_dual_pipe3">
  ];
}
def : PfmCountersBinding<"bdver2", BdVer2PfmCounters>;

def BtVer2PfmCounters : ProcPfmCounters {
  let CycleCounter = PfmCounter<"cpu_clk_unhalted">;
  let UopsCounter = PfmCounter<"retired_uops">;
  let IssueCounters = [
    PfmIssueCounter<"JFPU0", "dispatched_fpu:pipe0">,
    PfmIssueCounter<"JFPU1", "dispatched_fpu:pipe1">
  ];
}
def : PfmCountersBinding<"btver2", BtVer2PfmCounters>;
