; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck --check-prefix=CHECK --check-prefix=CHECK-STRQRO %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -mcpu=falkor | FileCheck --check-prefix=CHECK --check-prefix=CHECK-NOSTRQRO %s

; CHECK-LABEL: strqrox:
; CHECK-STRQRO: str q{{[0-9]+}}, [x{{[0-9]+}}, x
; CHECK-NOSTRQRO-NOT: str q{{[0-9]+}}, [x{{[0-9]+}}, x
define void @strqrox(fp128 %val64, i64 %base, i64 %offset) {
  %addrint = add i64 %base, %offset
  %addr = inttoptr i64 %addrint to fp128*
  store volatile fp128 %val64, fp128* %addr
  ret void
}

; Check that STRQro is generated for both cases if we're optimizing for code size.
; CHECK-LABEL: strqrox_optsize:
; CHECK-STRQRO: str q{{[0-9]+}}, [x{{[0-9]+}}, x
; CHECK-NOSTRQRO: str q{{[0-9]+}}, [x{{[0-9]+}}, x
define void @strqrox_optsize(fp128 %val64, i64 %base, i64 %offset) minsize {
  %addrint = add i64 %base, %offset
  %addr = inttoptr i64 %addrint to fp128*
  store volatile fp128 %val64, fp128* %addr
  ret void
}

; CHECK-LABEL: strqrow:
; CHECK-STRQRO: str q{{[0-9]+}}, [x{{[0-9]+}}, w
; CHECK-NOSTRQRO-NOT: str q{{[0-9]+}}, [x{{[0-9]+}}, w
define void @strqrow(fp128 %val64, i64 %base, i32 %offset) {
  %offset64 = zext i32 %offset to i64
  %addrint = add i64 %base, %offset64
  %addr = inttoptr i64 %addrint to fp128*
  store volatile fp128 %val64, fp128* %addr
  ret void
}

; Check that STRQro is generated for both cases if we're optimizing for code size.
; CHECK-LABEL: strqrow_optsize:
; CHECK-STRQRO: str q{{[0-9]+}}, [x{{[0-9]+}}, w
; CHECK-NOSTRQRO: str q{{[0-9]+}}, [x{{[0-9]+}}, w
define void @strqrow_optsize(fp128 %val64, i64 %base, i32 %offset) minsize {
  %offset64 = zext i32 %offset to i64
  %addrint = add i64 %base, %offset64
  %addr = inttoptr i64 %addrint to fp128*
  store volatile fp128 %val64, fp128* %addr
  ret void
}

