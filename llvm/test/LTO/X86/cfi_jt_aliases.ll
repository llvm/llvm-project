; Full LTO test
; RUN: opt %s -o %t.bc
; RUN: llvm-lto2 run -o %t.o %t.bc -save-temps \
; RUN:   -r=%t.bc,a,px \
; RUN:   -r=%t.bc,b,px \
; RUN:   -r=%t.bc,c,px \
; RUN:   -r=%t.bc,d,px
; RUN: llvm-readelf --symbols %t.o.0 | grep \.cfi_jt | FileCheck --check-prefix=CHECK-FULL-RE %s
; RUN: llvm-objdump -dr %t.o.0 | FileCheck --check-prefix=CHECK-FULL-OD %s
; RUN: llvm-dis %t.o.0.4.opt.bc -o - | FileCheck --check-prefix=CHECK-USED %s
; Thin LTO test
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %s -o %t.bc
; RUN: llvm-lto2 run -o %t.o %t.bc \
; RUN:   -r=%t.bc,a,px \
; RUN:   -r=%t.bc,b,px \
; RUN:   -r=%t.bc,c,px \
; RUN:   -r=%t.bc,d,px
; RUN: llvm-readelf --symbols %t.o.0 | grep \.cfi_jt | FileCheck --check-prefix=CHECK-THIN-RE %s
; RUN: llvm-objdump -dr %t.o.0 | FileCheck --check-prefix=CHECK-THIN-OD %s

; CHECK-FULL-RE:      FUNC LOCAL DEFAULT {{[0-9]+}} a.cfi_jt
; CHECK-FULL-RE-NEXT: FUNC LOCAL DEFAULT {{[0-9]+}} b.cfi_jt
; CHECK-FULL-RE-NEXT: FUNC LOCAL DEFAULT {{[0-9]+}} c.cfi_jt

; CHECK-THIN-RE:      FUNC GLOBAL HIDDEN {{[0-9]+}} b.cfi_jt
; CHECK-THIN-RE-NEXT: FUNC GLOBAL HIDDEN {{[0-9]+}} c.cfi_jt
; CHECK-THIN-RE-NEXT: FUNC GLOBAL HIDDEN {{[0-9]+}} a.cfi_jt

; CHECK-FULL-OD:      a.cfi_jt>:
; CHECK-FULL-OD:      jmp {{.*}} <a.cfi_jt
; CHECK-FULL-OD-NEXT: R_X86_64_PLT32 .La$local
; CHECK-FULL-OD:      b.cfi_jt>:
; CHECK-FULL-OD:      jmp {{.*}} <b.cfi_jt
; CHECK-FULL-OD-NEXT: R_X86_64_PLT32 .Lb$local
; CHECK-FULL-OD:      c.cfi_jt>:
; CHECK-FULL-OD:      jmp {{.*}} <c.cfi_jt
; CHECK-FULL-OD-NEXT: R_X86_64_PLT32 .Lc$local

; CHECK-THIN-OD:      b.cfi_jt>:
; CHECK-THIN-OD:      jmp {{.*}} <b.cfi_jt
; CHECK-THIN-OD-NEXT: R_X86_64_PLT32 b
; CHECK-THIN-OD:      c.cfi_jt>:
; CHECK-THIN-OD:      jmp {{.*}} <c.cfi_jt
; CHECK-THIN-OD-NEXT: R_X86_64_PLT32 c
; CHECK-THIN-OD:      a.cfi_jt>:
; CHECK-THIN-OD:      jmp {{.*}} <a.cfi_jt
; CHECK-THIN-OD-NEXT: R_X86_64_PLT32 a

; CHECK-USED: @llvm.used = appending global [3 x i8*] [i8* bitcast (void ()* @a.cfi_jt to i8*), i8* bitcast (void ()* @b.cfi_jt to i8*), i8* bitcast (void ()* @c.cfi_jt to i8*)], section "llvm.metadata"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@f = internal global [4 x void ()*] [void ()* @a, void ()* @b, void ()* @c, void ()* null], align 16

define dso_local void @a() !type !5 !type !6 { ret void }
define dso_local void @b() !type !5 !type !6 { ret void }
define dso_local void @c() !type !5 !type !6 { ret void }

define dso_local void @d() !type !5 !type !6 {
entry:
  %i = alloca i32, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds [4 x void ()*], [4 x void ()*]* @f, i64 0, i64 %idxprom
  %1 = load void ()*, void ()** %arrayidx, align 8
  %tobool = icmp ne void ()* %1, null
  br i1 %tobool, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %i, align 4
  %idxprom1 = sext i32 %2 to i64
  %arrayidx2 = getelementptr inbounds [4 x void ()*], [4 x void ()*]* @f, i64 0, i64 %idxprom1
  %3 = load void ()*, void ()** %arrayidx2, align 8
  %4 = bitcast void ()* %3 to i8*, !nosanitize !7
  %5 = call i1 @llvm.type.test(i8* %4, metadata !"_ZTSFvvE"), !nosanitize !7
  br i1 %5, label %cont, label %trap, !nosanitize !7

trap:                                             ; preds = %for.body
  call void @llvm.ubsantrap(i8 2), !nosanitize !7
  unreachable, !nosanitize !7

cont:                                             ; preds = %for.body
  call void %3()
  br label %for.inc

for.inc:                                          ; preds = %cont
  %6 = load i32, i32* %i, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.ubsantrap(i8 immarg)

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"CFI Canonical Jump Tables", i32 0}
!2 = !{i32 7, !"uwtable", i32 1}
!3 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{i64 0, !"_ZTSFvvE"}
!6 = !{i64 0, !"_ZTSFvvE.generalized"}
!7 = !{}
