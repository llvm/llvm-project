; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s -check-prefix=COLDCC

%struct.MyStruct = type { i32, i32, i32, i32 }

@caller.s = internal unnamed_addr global %struct.MyStruct zeroinitializer, align 8

define signext i32 @caller(i32 signext %a, i32 signext %b, i32 signext %cold) {
entry:
; COLDCC: bl callee
; COLDCC: ld 4, 40(1)
; COLDCC: ld 5, 32(1)
  %call = tail call coldcc { i64, i64 } @callee(i32 signext %a, i32 signext %b)
  %0 = extractvalue { i64, i64 } %call, 0
  %1 = extractvalue { i64, i64 } %call, 1
  store i64 %0, ptr @caller.s, align 8
  store i64 %1, ptr getelementptr inbounds (%struct.MyStruct, ptr @caller.s, i64 0, i32 2), align 8
  %2 = lshr i64 %1, 32
  %3 = trunc i64 %2 to i32
  %sub = sub nsw i32 0, %3
  ret i32 %sub
}

define internal coldcc { i64, i64 } @callee(i32 signext %a, i32 signext %b) {
entry:
; COLDCC: std {{[0-9]+}}, 0(3)
; COLDCC: std {{[0-9]+}}, 8(3)
  %0 = tail call i32 asm "add $0, $1, $2", "=r,r,r,~{r6},~{r7},~{r8},~{r9},~{r10}"(i32 %a, i32 %b)
  %mul = mul nsw i32 %a, 3
  %1 = mul i32 %b, -5
  %add = add i32 %1, %mul
  %sub = add i32 %add, %0
  %mul5 = mul nsw i32 %b, %a
  %add6 = add nsw i32 %sub, %mul5
  %retval.sroa.0.0.insert.ext = zext i32 %0 to i64
  %retval.sroa.3.8.insert.ext = zext i32 %sub to i64
  %retval.sroa.3.12.insert.ext = zext i32 %add6 to i64
  %retval.sroa.3.12.insert.shift = shl nuw i64 %retval.sroa.3.12.insert.ext, 32
  %retval.sroa.3.12.insert.insert = or i64 %retval.sroa.3.12.insert.shift, %retval.sroa.3.8.insert.ext
  %.fca.0.insert = insertvalue { i64, i64 } undef, i64 %retval.sroa.0.0.insert.ext, 0
  %.fca.1.insert = insertvalue { i64, i64 } %.fca.0.insert, i64 %retval.sroa.3.12.insert.insert, 1
  ret { i64, i64 } %.fca.1.insert
}
