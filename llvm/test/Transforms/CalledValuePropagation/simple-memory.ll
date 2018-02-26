; RUN: opt -called-value-propagation -S < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnueabi"

@global_function = internal unnamed_addr global void ()* null, align 8
@global_array = common unnamed_addr global i64* null, align 8

; This test checks that we propagate the functions through an internal global
; variable, and attach !callees metadata to the call. Such metadata can enable
; optimizations of this code sequence.
;
; For example, since both of the targeted functions have the "nounwind" and
; "readnone" function attributes, LICM can be made to move the call and the
; function pointer load outside the loop. This would then enable the loop
; vectorizer to vectorize the sum reduction.
;
; CHECK: call void %tmp0(), !callees ![[MD:[0-9]+]]
; CHECK: ![[MD]] = !{void ()* @invariant_1, void ()* @invariant_2}
;
define i64 @test_memory_entry(i64 %n, i1 %flag) {
entry:
  br i1 %flag, label %then, label %else

then:
  store void ()* @invariant_1, void ()** @global_function
  br label %merge

else:
  store void ()* @invariant_2, void ()** @global_function
  br label %merge

merge:
  %tmp1 = call i64 @test_memory(i64 %n)
  ret i64 %tmp1
}

define internal i64 @test_memory(i64 %n) {
entry:
  %array = load i64*, i64** @global_array
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %r = phi i64 [ 0, %entry ], [ %tmp3, %for.body ]
  %tmp0 = load void ()*, void ()** @global_function
  call void %tmp0()
  %tmp1 = getelementptr inbounds i64, i64* %array, i64 %i
  %tmp2 = load i64, i64* %tmp1
  %tmp3 = add i64 %tmp2, %r
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp4 = phi i64 [ %tmp3, %for.body ]
  ret i64 %tmp4
}

declare void @invariant_1() #0
declare void @invariant_2() #0

attributes #0 = { nounwind readnone }
