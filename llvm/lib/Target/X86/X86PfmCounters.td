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

let SchedModel = SandyBridgeModel in {
def SBCycleCounter : PfmCycleCounter<"unhalted_core_cycles">;
def SBPort0Counter : PfmIssueCounter<SBPort0, ["uops_dispatched_port:port_0"]>;
def SBPort1Counter : PfmIssueCounter<SBPort1, ["uops_dispatched_port:port_1"]>;
def SBPort23Counter : PfmIssueCounter<SBPort23,
                                      ["uops_dispatched_port:port_2",
                                       "uops_dispatched_port:port_3"]>;
def SBPort4Counter : PfmIssueCounter<SBPort4, ["uops_dispatched_port:port_4"]>;
def SBPort5Counter : PfmIssueCounter<SBPort5, ["uops_dispatched_port:port_5"]>;
}

let SchedModel = HaswellModel in {
def HWCycleCounter : PfmCycleCounter<"unhalted_core_cycles">;
def HWPort0Counter : PfmIssueCounter<HWPort0, ["uops_dispatched_port:port_0"]>;
def HWPort1Counter : PfmIssueCounter<HWPort1, ["uops_dispatched_port:port_1"]>;
def HWPort2Counter : PfmIssueCounter<HWPort2, ["uops_dispatched_port:port_2"]>;
def HWPort3Counter : PfmIssueCounter<HWPort3, ["uops_dispatched_port:port_3"]>;
def HWPort4Counter : PfmIssueCounter<HWPort4, ["uops_dispatched_port:port_4"]>;
def HWPort5Counter : PfmIssueCounter<HWPort5, ["uops_dispatched_port:port_5"]>;
def HWPort6Counter : PfmIssueCounter<HWPort6, ["uops_dispatched_port:port_6"]>;
def HWPort7Counter : PfmIssueCounter<HWPort7, ["uops_dispatched_port:port_7"]>;
}

let SchedModel = BroadwellModel in {
def BWCycleCounter : PfmCycleCounter<"unhalted_core_cycles">;
def BWPort0Counter : PfmIssueCounter<BWPort0, ["uops_executed_port:port_0"]>;
def BWPort1Counter : PfmIssueCounter<BWPort1, ["uops_executed_port:port_1"]>;
def BWPort2Counter : PfmIssueCounter<BWPort2, ["uops_executed_port:port_2"]>;
def BWPort3Counter : PfmIssueCounter<BWPort3, ["uops_executed_port:port_3"]>;
def BWPort4Counter : PfmIssueCounter<BWPort4, ["uops_executed_port:port_4"]>;
def BWPort5Counter : PfmIssueCounter<BWPort5, ["uops_executed_port:port_5"]>;
def BWPort6Counter : PfmIssueCounter<BWPort6, ["uops_executed_port:port_6"]>;
def BWPort7Counter : PfmIssueCounter<BWPort7, ["uops_executed_port:port_7"]>;
}

let SchedModel = SkylakeClientModel in {
def SKLCycleCounter : PfmCycleCounter<"unhalted_core_cycles">;
def SKLPort0Counter : PfmIssueCounter<SKLPort0, ["uops_dispatched_port:port_0"]>;
def SKLPort1Counter : PfmIssueCounter<SKLPort1, ["uops_dispatched_port:port_1"]>;
def SKLPort2Counter : PfmIssueCounter<SKLPort2, ["uops_dispatched_port:port_2"]>;
def SKLPort3Counter : PfmIssueCounter<SKLPort3, ["uops_dispatched_port:port_3"]>;
def SKLPort4Counter : PfmIssueCounter<SKLPort4, ["uops_dispatched_port:port_4"]>;
def SKLPort5Counter : PfmIssueCounter<SKLPort5, ["uops_dispatched_port:port_5"]>;
def SKLPort6Counter : PfmIssueCounter<SKLPort6, ["uops_dispatched_port:port_6"]>;
def SKLPort7Counter : PfmIssueCounter<SKLPort7, ["uops_dispatched_port:port_7"]>;
}

let SchedModel = SkylakeServerModel in {
def SKXCycleCounter : PfmCycleCounter<"unhalted_core_cycles">;
def SKXPort0Counter : PfmIssueCounter<SKXPort0, ["uops_dispatched_port:port_0"]>;
def SKXPort1Counter : PfmIssueCounter<SKXPort1, ["uops_dispatched_port:port_1"]>;
def SKXPort2Counter : PfmIssueCounter<SKXPort2, ["uops_dispatched_port:port_2"]>;
def SKXPort3Counter : PfmIssueCounter<SKXPort3, ["uops_dispatched_port:port_3"]>;
def SKXPort4Counter : PfmIssueCounter<SKXPort4, ["uops_dispatched_port:port_4"]>;
def SKXPort5Counter : PfmIssueCounter<SKXPort5, ["uops_dispatched_port:port_5"]>;
def SKXPort6Counter : PfmIssueCounter<SKXPort6, ["uops_dispatched_port:port_6"]>;
def SKXPort7Counter : PfmIssueCounter<SKXPort7, ["uops_dispatched_port:port_7"]>;
}

let SchedModel = BtVer2Model in {
def JCycleCounter : PfmCycleCounter<"cpu_clk_unhalted">;
def JFPU0Counter  : PfmIssueCounter<JFPU0, ["dispatched_fpu:pipe0"]>;
def JFPU1Counter  : PfmIssueCounter<JFPU1, ["dispatched_fpu:pipe1"]>;
}
