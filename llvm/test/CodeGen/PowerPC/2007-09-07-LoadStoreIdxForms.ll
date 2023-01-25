; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -O1 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- | FileCheck --check-prefix=CHECK-OPT  %s

        %struct.__db_region = type { %struct.__mutex_t, [4 x i8], %struct.anon, i32, [1 x i32] }
        %struct.__mutex_t = type { i32 }
        %struct.anon = type { i64, i64 }

define void @foo() {
entry:
        %ttype = alloca i32, align 4            ; <ptr> [#uses=1]
        %regs = alloca [1024 x %struct.__db_region], align 16           ; <ptr> [#uses=0]
        %tmp = load i32, ptr %ttype, align 4                ; <i32> [#uses=1]
        %tmp1 = call i32 (...) @bork( i32 %tmp )               ; <i32> [#uses=0]
        ret void

; CHECK: @foo
; CHECK: lwzx
; CHECK: blr
; CHECK-OPT: @foo
; CHECK-OPT: lwz
; CHECK-OPT: blr
}

define signext i32 @test(ptr noalias nocapture readonly %b, i32 signext %n)  {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %mul = mul nsw i32 %0, 7
  ret i32 %mul

; CHECK-OPT: @test
; CHECK-OPT: lwzx
; CHECK-OPT: blr

}


declare i32 @bork(...)
