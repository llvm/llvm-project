; RUN: llvm-reduce -j=1 --abort-on-invalid-reduction \
; RUN:   --delta-passes=operands-zero \
; RUN:   -o %t.reduced.ll %s \
; RUN:   --test=%python --test-arg %p/Inputs/llvm-as-and-filecheck.py \
; RUN:   --test-arg llvm-as \
; RUN:   --test-arg FileCheck --test-arg --check-prefix=INTERESTING \
; RUN:   --test-arg %s

; Check if the final output really parses
; RUN: not llvm-as -o /dev/null %t.reduced.ll
; RUN: FileCheck --check-prefix=RESULT %s < %t.reduced.ll


define void @kernel_ocl_path_trace_direct_lighting(i1 %cond.i, i1 %cmp5.i.i, i32 %arg) {
; INTERESTING: entry:
; INTERESTING: 0
; INTERESTING: 0
; INTERESTING: %cmp5.i.i2 = icmp slt i32 {{[0-9]+}}, 0
entry:
  %add_zero_a = add i32 %arg, 0
  %load1.i1 = load i32, ptr addrspace(1) null, align 4
  %add_zero_b = add i32 %arg, 0
  %cmp5.i.i2 = icmp slt i32 1, 0
  br i1 %cond.i, label %if.end13.i.i, label %if.then6.i.i

; INTERESTING: if.then6.i.i:
; INTERESTING: %cond.i4 = icmp eq i32 %load0.i, 0
if.then6.i.i:
  %load0.i = load i32, ptr addrspace(4) null, align 4
  %cond.i4 = icmp eq i32 %load0.i, 0
  %extractVec358.i.i = insertelement <4 x float> zeroinitializer, float 1.000000e+00, i64 0
  br i1 %cmp5.i.i, label %if.end13.i.i, label %kernel_direct_lighting.exit

if.end13.i.i:
  br i1 false, label %if.then263.i.i, label %if.end273.i.i

; INTERESTING: if.then263.i.i:
; INTERESTING-NEXT: i32 0
if.then263.i.i:
  %extractVec72.i.i.i11 = shufflevector <3 x float> zeroinitializer, <3 x float> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  br i1 %cond.i, label %if.end273.i.i, label %kernel_direct_lighting.exit

; INTERESTING: if.end273.i.i:
if.end273.i.i:
  br label %kernel_direct_lighting.exit

kernel_direct_lighting.exit:
  ret void
}

; RESULT: uselistorder i32 0, { 4, 0, 5, 1, 6, 2, 7, 3 }
