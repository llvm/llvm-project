; RUN: opt -inline -S -o - < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S < %s | FileCheck %s

%0 = type opaque
%struct.Foo = type { i32, %0* }

; Test that we don't crash when inlining @bar (rdar://22521387).
define void @foo(%struct.Foo* align 4 %a) {
entry:
  call fastcc void @bar(%struct.Foo* nonnull align 4 undef)

; CHECK: call void @llvm.assume(i1 undef)
; CHECK: unreachable

  ret void
}

define fastcc void @bar(%struct.Foo* align 4 %a) {
; CHECK-LABEL: @bar
entry:
  %b = getelementptr inbounds %struct.Foo, %struct.Foo* %a, i32 0, i32 1
  br i1 undef, label %if.end, label %if.then.i.i

if.then.i.i:
  call void @llvm.assume(i1 undef)
  unreachable

if.end:
  ret void
}

declare void @llvm.assume(i1)
