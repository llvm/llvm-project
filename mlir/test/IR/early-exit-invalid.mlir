
// RUN: mlir-opt %s --split-input-file --verify-diagnostics


// expected-error @+1 {{operation has a nested predessor but does not have the HasBreakingControlFlowOpInterface trait}}
  func.func @loop_continue() {
   scf.loop {
// expected-note @+1 {{for this predecessor operation (scf.continue)}}
     scf.continue 2
   } loc("loop1")
   return
}

// -----

func.func @loop_result_mismatch(%value : f32) {
 // expected-error @+1 {{'scf.loop' op along control flow edge from Operation scf.break to parent results: source type #0 'f32' should match input type #0 'i32'}}
 %result = scf.loop -> i32 {
   scf.break 1 %value : f32
 }
 return
}

// -----

func.func @loop_result_number_mismatch(%value : f32) {
 // expected-error @+1 {{'scf.loop' op region control flow edge from Operation scf.break to parent results: source has 1 operands, but target successor <to parent> needs 2}}
 %result:2 = scf.loop -> f32, f32 {
   scf.break 1 %value : f32
 }
 return
}

// -----

func.func @loop_continue_mismatch(%init : i32, %value : f32) {
 // expected-error @+1 {{'scf.loop' op along control flow edge from Operation scf.continue to Region #0: source type #0 'f32' should match input type #0 'i32'}}
 scf.loop iter_args(%next = %init) : i32 {
   scf.continue 1 %value : f32
 }
 return
}


// -----

func.func @loop_iterargs_mismatch(%init : i32, %value : f32) {
 // expected-error @+1 {{'scf.loop' op along control flow edge from Operation scf.continue to Region #0: source type #0 'i32' should match input type #0 'f32'}}
 "scf.loop"(%init) ({
    ^body(%next : f32):
   scf.continue 1 %init : i32
 })  : (i32) -> ()
 return
}

// -----

func.func @loop_iterargs_mismatch(%init : i32, %value : f32) {
 // expected-error @+1 {{'scf.loop' op region control flow edge from Operation scf.continue to Region #0: source has 1 operands, but target successor <to region #0 with 2 inputs> needs 2}}
 "scf.loop"(%init) ({
    ^body(%next : i32, %next2 : f32):
   scf.continue 1 %init : i32
 })  : (i32) -> ()
 return
}
