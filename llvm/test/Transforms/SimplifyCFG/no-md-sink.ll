; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -sink-common-insts -S | FileCheck %s
; RUN: opt < %s -passes='simplifycfg<sink-common-insts>' -S | FileCheck %s

define i1 @test1(i1 zeroext %flag, i8* %y) #0 {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %r = call i1 @llvm.type.test(i8* %y, metadata !0)
  br label %if.end

if.else:
  %s = call i1 @llvm.type.test(i8* %y, metadata !1)
  br label %if.end

if.end:
  %t = phi i1 [ %s, %if.else ], [ %r, %if.then ]
  ret i1 %t
}

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 4, !"typeid1"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

; CHECK-LABEL: test1
; CHECK: @llvm.type.test
; CHECK: @llvm.type.test
; CHECK: ret i1

define i1 @test2(i1 zeroext %flag, i8* %y, i8* %z) #0 {
entry:
  br i1 %flag, label %if.then, label %if.else

if.then:
  %r = call i1 @llvm.type.test(i8* %y, metadata !0)
  br label %if.end

if.else:
  %s = call i1 @llvm.type.test(i8* %z, metadata !0)
  br label %if.end

if.end:
  %t = phi i1 [ %s, %if.else ], [ %r, %if.then ]
  ret i1 %t
}

; CHECK-LABEL: test2
; CHECK: %[[S:[a-z0-9.]+]] = select i1 %flag, i8* %y, i8* %z
; CHECK: %[[R:[a-z0-9.]+]] = call i1 @llvm.type.test(i8* %[[S]], metadata ![[MD:[0-9]+]]
; CHECK: ret i1 %[[R]]
; CHECK: ![[MD]] = !{i32 0, !"typeid1"}
