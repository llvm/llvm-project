; RUN: llc < %s -march=xcore | FileCheck %s

%struct.FRAME.f = type { i32, i32 ()* }

define void @f() nounwind {
entry:
; CHECK: f:
; CHECK: ldap r11, g.1101
; CHECK: stw r11, sp[7]
  %TRAMP.23 = alloca [20 x i8], align 2
  %FRAME.0 = alloca %struct.FRAME.f, align 4
  %TRAMP.23.sub = getelementptr inbounds [20 x i8]* %TRAMP.23, i32 0, i32 0
  %FRAME.02 = bitcast %struct.FRAME.f* %FRAME.0 to i8*
  call void @llvm.init.trampoline(i8* %TRAMP.23.sub, i8* bitcast (i32 (%struct.FRAME.f*)* @g.1101 to i8*), i8* %FRAME.02)
  %tramp = call i8* @llvm.adjust.trampoline(i8* %TRAMP.23.sub)
  %0 = getelementptr inbounds %struct.FRAME.f* %FRAME.0, i32 0, i32 1
  %1 = bitcast i8* %tramp to i32 ()*
  store i32 ()* %1, i32 ()** %0, align 4
  %2 = getelementptr inbounds %struct.FRAME.f* %FRAME.0, i32 0, i32 0
  store i32 1, i32* %2, align 4
  call void @h(i32 ()* %1) nounwind
  ret void
}

define internal i32 @g.1101(%struct.FRAME.f* nocapture nest %CHAIN.1) nounwind readonly {
entry:
; CHECK: g.1101:
; CHECK: ldw r11, sp[0]
; CHECK-NEXT: ldw r0, r11[0]
; CHECK-NEXT: retsp 0
  %0 = getelementptr inbounds %struct.FRAME.f* %CHAIN.1, i32 0, i32 0
  %1 = load i32* %0, align 4
  ret i32 %1
}

declare void @llvm.init.trampoline(i8*, i8*, i8*) nounwind
declare i8* @llvm.adjust.trampoline(i8*) nounwind

declare void @h(i32 ()*)
