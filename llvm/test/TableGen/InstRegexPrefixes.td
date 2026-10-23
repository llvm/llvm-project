// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s | FileCheck %s

// Check the instructions matched by instregex patterns whose literal prefix
// is used to narrow the search.

include "llvm/Target/Target.td"

def MyTarget : Target;

let OutOperandList = (outs), InOperandList = (ins) in {
  def ABD : Instruction;
  def LDPSW : Instruction;
  def LDPW : Instruction;
  def LDPX : Instruction;
  def SABD : Instruction;
  def SABDL : Instruction;
  def SCVTFa : Instruction;
  def SCVTFb : Instruction;
  def SM : Instruction;
  def SUM : Instruction;
  def SUMX : Instruction;
  def UABD : Instruction;
  def UCVTFa : Instruction;
  def XCVTF : Instruction;
}

class Model : SchedMachineModel {
  let CompleteModel = 0;
}

// A group of literal alternatives at the start.
def ModelGroup : Model;
let SchedModel = ModelGroup in {
  def WriteGroup : SchedWriteRes<[]>;
  def : InstRW<[WriteGroup], (instregex "^(S|U)CVTF")>;
}

// A character class at the start, followed by a quantified character.
def ModelClass : Model;
let SchedModel = ModelClass in {
  def WriteClass : SchedWriteRes<[]>;
  def : InstRW<[WriteClass], (instregex "[SU]ABDL?$")>;
}

// Overlapping alternatives.
def ModelOverlap : Model;
let SchedModel = ModelOverlap in {
  def WriteOverlap : SchedWriteRes<[]>;
  def : InstRW<[WriteOverlap], (instregex "^(S|SU)M")>;
}

// A quantified group can't be used as a prefix.
def ModelOptional : Model;
let SchedModel = ModelOptional in {
  def WriteOptional : SchedWriteRes<[]>;
  def : InstRW<[WriteOptional], (instregex "^(S|U)?ABD$")>;
}

// A top-level ? after the prefix.
def ModelQuestion : Model;
let SchedModel = ModelQuestion in {
  def WriteQuestion : SchedWriteRes<[]>;
  def : InstRW<[WriteQuestion], (instregex "LDPS?W")>;
}

def ProcGroup : ProcessorModel<"group", ModelGroup, []>;
def ProcClass : ProcessorModel<"class", ModelClass, []>;
def ProcOverlap : ProcessorModel<"overlap", ModelOverlap, []>;
def ProcOptional : ProcessorModel<"optional", ModelOptional, []>;
def ProcQuestion : ProcessorModel<"question", ModelQuestion, []>;

// CHECK-LABEL: static const llvm::MCSchedClassDesc ModelClassSchedClasses[] = {{{$}}
// CHECK-NEXT: {DBGFIELD(1) 8191, false, false, false, 0, 0, 0, 0, 0, 0},{{$}}
// CHECK-NEXT: {DBGFIELD(/*SCVTFa_SCVTFb_UCVTFa*/ 19) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #1{{$}}
// CHECK-NEXT: {DBGFIELD(/*SABD_SABDL_UABD*/ 40) 1, false, false, false, 0, 0, 1, 0, 0, 1}, // #2{{$}}
// CHECK-NEXT: {DBGFIELD(/*SM_SUM_SUMX*/ 56) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #3{{$}}
// CHECK-NEXT: {DBGFIELD(/*ABD*/ 68) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #4{{$}}
// CHECK-NEXT: {DBGFIELD(/*SABD_UABD*/ 72) 1, false, false, false, 0, 0, 1, 0, 0, 1}, // #5{{$}}
// CHECK-NEXT: {DBGFIELD(/*LDPSW_LDPW*/ 82) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #6{{$}}

// CHECK-LABEL: static const llvm::MCSchedClassDesc ModelGroupSchedClasses[] = {{{$}}
// CHECK-NEXT: {DBGFIELD(1) 8191, false, false, false, 0, 0, 0, 0, 0, 0},{{$}}
// CHECK-NEXT: {DBGFIELD(/*SCVTFa_SCVTFb_UCVTFa*/ 19) 1, false, false, false, 0, 0, 1, 0, 0, 1}, // #1{{$}}
// CHECK-NEXT: {DBGFIELD(/*SABD_SABDL_UABD*/ 40) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #2{{$}}
// CHECK-NEXT: {DBGFIELD(/*SM_SUM_SUMX*/ 56) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #3{{$}}
// CHECK-NEXT: {DBGFIELD(/*ABD*/ 68) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #4{{$}}
// CHECK-NEXT: {DBGFIELD(/*SABD_UABD*/ 72) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #5{{$}}
// CHECK-NEXT: {DBGFIELD(/*LDPSW_LDPW*/ 82) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #6{{$}}

// CHECK-LABEL: static const llvm::MCSchedClassDesc ModelOptionalSchedClasses[] = {{{$}}
// CHECK-NEXT: {DBGFIELD(1) 8191, false, false, false, 0, 0, 0, 0, 0, 0},{{$}}
// CHECK-NEXT: {DBGFIELD(/*SCVTFa_SCVTFb_UCVTFa*/ 19) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #1{{$}}
// CHECK-NEXT: {DBGFIELD(/*SABD_SABDL_UABD*/ 40) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #2{{$}}
// CHECK-NEXT: {DBGFIELD(/*SM_SUM_SUMX*/ 56) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #3{{$}}
// CHECK-NEXT: {DBGFIELD(/*ABD*/ 68) 1, false, false, false, 0, 0, 1, 0, 0, 1}, // #4{{$}}
// CHECK-NEXT: {DBGFIELD(/*SABD_UABD*/ 72) 1, false, false, false, 0, 0, 1, 0, 0, 1}, // #5{{$}}
// CHECK-NEXT: {DBGFIELD(/*LDPSW_LDPW*/ 82) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #6{{$}}

// CHECK-LABEL: static const llvm::MCSchedClassDesc ModelOverlapSchedClasses[] = {{{$}}
// CHECK-NEXT: {DBGFIELD(1) 8191, false, false, false, 0, 0, 0, 0, 0, 0},{{$}}
// CHECK-NEXT: {DBGFIELD(/*SCVTFa_SCVTFb_UCVTFa*/ 19) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #1{{$}}
// CHECK-NEXT: {DBGFIELD(/*SABD_SABDL_UABD*/ 40) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #2{{$}}
// CHECK-NEXT: {DBGFIELD(/*SM_SUM_SUMX*/ 56) 1, false, false, false, 0, 0, 1, 0, 0, 1}, // #3{{$}}
// CHECK-NEXT: {DBGFIELD(/*ABD*/ 68) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #4{{$}}
// CHECK-NEXT: {DBGFIELD(/*SABD_UABD*/ 72) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #5{{$}}
// CHECK-NEXT: {DBGFIELD(/*LDPSW_LDPW*/ 82) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #6{{$}}

// CHECK-LABEL: static const llvm::MCSchedClassDesc ModelQuestionSchedClasses[] = {{{$}}
// CHECK-NEXT: {DBGFIELD(1) 8191, false, false, false, 0, 0, 0, 0, 0, 0},{{$}}
// CHECK-NEXT: {DBGFIELD(/*SCVTFa_SCVTFb_UCVTFa*/ 19) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #1{{$}}
// CHECK-NEXT: {DBGFIELD(/*SABD_SABDL_UABD*/ 40) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #2{{$}}
// CHECK-NEXT: {DBGFIELD(/*SM_SUM_SUMX*/ 56) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #3{{$}}
// CHECK-NEXT: {DBGFIELD(/*ABD*/ 68) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #4{{$}}
// CHECK-NEXT: {DBGFIELD(/*SABD_UABD*/ 72) 8191, false, false, false, 0, 0, 0, 0, 0, 0}, // #5{{$}}
// CHECK-NEXT: {DBGFIELD(/*LDPSW_LDPW*/ 82) 1, false, false, false, 0, 0, 1, 0, 0, 1}, // #6{{$}}
