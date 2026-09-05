; Verify that the ThinLTO cache key includes the contents of consumed CGData.
;
; RUN: rm -rf %t; split-file %s %t
; RUN: opt -module-summary -module-hash %t/foo.ll -o %t/foo.bc
; RUN: opt -module-summary -module-hash %t/goo.ll -o %t/goo.bc
; RUN: opt -module-summary -module-hash %t/x.ll -o %t/x.bc
; RUN: opt -module-summary -module-hash %t/y.ll -o %t/y.bc
;
; Generate two valid CGData files containing different function maps.
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -codegen-data-generate=true %t/foo.bc %t/goo.bc -o %t/a-write \
; RUN:   -r %t/foo.bc,_f1,px -r %t/goo.bc,_f2,px \
; RUN:   -r %t/foo.bc,_g,l -r %t/foo.bc,_g1,l \
; RUN:   -r %t/goo.bc,_g,l -r %t/goo.bc,_g2,l
; RUN: llvm-cgdata --merge -o %t/a.cgdata %t/a-write.1 %t/a-write.2
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -codegen-data-generate=true %t/x.bc %t/y.bc -o %t/b-write \
; RUN:   -r %t/x.bc,_x1,px -r %t/y.bc,_x2,px \
; RUN:   -r %t/x.bc,_h,l -r %t/x.bc,_h1,l \
; RUN:   -r %t/y.bc,_h,l -r %t/y.bc,_h2,l
; RUN: llvm-cgdata --merge -o %t/b.cgdata %t/b-write.1 %t/b-write.2
;
; Populate a cache while the stable path contains the map for f1/f2.
; RUN: cp %t/a.cgdata %t/active.cgdata
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -codegen-data-use-path=%t/active.cgdata -cache-dir=%t/cache \
; RUN:   %t/foo.bc %t/goo.bc -o %t/use-a \
; RUN:   -r %t/foo.bc,_f1,px -r %t/goo.bc,_f2,px \
; RUN:   -r %t/foo.bc,_g,l -r %t/foo.bc,_g1,l \
; RUN:   -r %t/goo.bc,_g,l -r %t/goo.bc,_g2,l
; RUN: llvm-nm %t/use-a.1 | FileCheck %s --check-prefix=MERGED
;
; Change only the contents at that path. The shared-cache result must match a
; fresh-cache run and must not reuse the f1/f2 merge result.
; RUN: cp %t/b.cgdata %t/active.cgdata
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -codegen-data-use-path=%t/active.cgdata -cache-dir=%t/cache \
; RUN:   %t/foo.bc %t/goo.bc -o %t/use-b-shared \
; RUN:   -r %t/foo.bc,_f1,px -r %t/goo.bc,_f2,px \
; RUN:   -r %t/foo.bc,_g,l -r %t/foo.bc,_g1,l \
; RUN:   -r %t/goo.bc,_g,l -r %t/goo.bc,_g2,l
; RUN: llvm-lto2 run -enable-global-merge-func=true \
; RUN:   -codegen-data-use-path=%t/active.cgdata -cache-dir=%t/fresh-cache \
; RUN:   %t/foo.bc %t/goo.bc -o %t/use-b-fresh \
; RUN:   -r %t/foo.bc,_f1,px -r %t/goo.bc,_f2,px \
; RUN:   -r %t/foo.bc,_g,l -r %t/foo.bc,_g1,l \
; RUN:   -r %t/goo.bc,_g,l -r %t/goo.bc,_g2,l
; RUN: llvm-nm %t/use-b-shared.1 | FileCheck %s --check-prefix=UNMERGED
; RUN: cmp %t/use-b-shared.1 %t/use-b-fresh.1
; RUN: cmp %t/use-b-shared.2 %t/use-b-fresh.2
;
; MERGED: _f1.Tgm
; UNMERGED-NOT: _f1.Tgm
;
;--- foo.ll
source_filename = "foo.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios12.0.0"

@g = external local_unnamed_addr global [0 x i32], align 4
@g1 = external global i32, align 4

define i32 @f1(i32 %a) {
entry:
  %idx = sext i32 %a to i64
  %p = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idx
  %v = load i32, ptr %p, align 4
  %c = load volatile i32, ptr @g1, align 4
  %m = mul nsw i32 %c, %v
  %r = add nsw i32 %m, 1
  ret i32 %r
}

;--- goo.ll
source_filename = "goo.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios12.0.0"

@g = external local_unnamed_addr global [0 x i32], align 4
@g2 = external global i32, align 4

define i32 @f2(i32 %a) {
entry:
  %idx = sext i32 %a to i64
  %p = getelementptr inbounds [0 x i32], ptr @g, i64 0, i64 %idx
  %v = load i32, ptr %p, align 4
  %c = load volatile i32, ptr @g2, align 4
  %m = mul nsw i32 %c, %v
  %r = add nsw i32 %m, 1
  ret i32 %r
}

;--- x.ll
source_filename = "x.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios12.0.0"

@h = external local_unnamed_addr global [0 x i32], align 4
@h1 = external global i32, align 4

define i32 @x1(i32 %a) {
entry:
  %idx = sext i32 %a to i64
  %p = getelementptr inbounds [0 x i32], ptr @h, i64 0, i64 %idx
  %v = load i32, ptr %p, align 4
  %c = load volatile i32, ptr @h1, align 4
  %m = mul nsw i32 %c, %v
  %r = sub nsw i32 %m, 1
  ret i32 %r
}

;--- y.ll
source_filename = "y.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios12.0.0"

@h = external local_unnamed_addr global [0 x i32], align 4
@h2 = external global i32, align 4

define i32 @x2(i32 %a) {
entry:
  %idx = sext i32 %a to i64
  %p = getelementptr inbounds [0 x i32], ptr @h, i64 0, i64 %idx
  %v = load i32, ptr %p, align 4
  %c = load volatile i32, ptr @h2, align 4
  %m = mul nsw i32 %c, %v
  %r = sub nsw i32 %m, 1
  ret i32 %r
}
