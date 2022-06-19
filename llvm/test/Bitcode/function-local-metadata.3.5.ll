; RUN: llvm-dis -opaque-pointers=0 < %s.bc | FileCheck %s
; RUN: llvm-dis -opaque-pointers=1 < %s.bc | FileCheck %s

; Check that function-local metadata is dropped correctly when it's not a
; direct argument to a call instruction.
;
; Bitcode assembled by llvm-as v3.5.0.

define void @foo(i32 %v) {
; CHECK: entry:
entry:
; CHECK-NEXT: call void @llvm.bar(metadata i32 %v)
  call void @llvm.bar(metadata !{i32 %v})

; Note: these supposedly legal instructions fired an assertion in llvm-as:
;
; Assertion failed: (I != ValueMap.end() && "Value not in slotcalculator!"), function getValueID, file lib/Bitcode/Writer/ValueEnumerator.cpp, line 138.
;
; So, I didn't test them; it looks like bitcode compatability is irrelevant.
  ; call void @llvm.bar(metadata !{i32 0, i32 %v})
  ; call void @llvm.bar(metadata !{i32 %v, i32 0})
  ; call void @llvm.bar(metadata !{metadata !{}, i32 %v})
  ; call void @llvm.bar(metadata !{i32 %v, metadata !{}})

; CHECK-NEXT: call void @llvm.bar(metadata !0)
; CHECK-NEXT: call void @llvm.bar(metadata !0)
  call void @llvm.bar(metadata !{i32 %v, i32 %v})
  call void @llvm.bar(metadata !{metadata !{i32 %v}})

; CHECK-NEXT: ret void{{$}}
  ret void, !baz !{i32 %v}
}

declare void @llvm.bar(metadata)

; CHECK: !0 = !{}
