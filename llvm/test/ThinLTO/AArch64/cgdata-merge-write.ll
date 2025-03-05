; This test verifies whether a stable function is encoded into the __llvm_merge section
; when the -codegen-data-generate flag is used under -enable-global-merge-func=true.

; RUN: rm -rf %t; split-file %s %t

; RUN: opt -module-summary -module-hash %t/foo.ll -o %t-foo.bc
; RUN: opt -module-summary -module-hash %t/goo.ll -o %t-goo.bc

; RUN: llvm-lto2 run -enable-global-merge-func=true -codegen-data-generate=false %t-foo.bc %t-goo.bc -o %tout-nowrite \
; RUN:    -r %t-foo.bc,_f1,px \
; RUN:    -r %t-goo.bc,_f2,px \
; RUN:    -r %t-foo.bc,_g,l -r %t-foo.bc,_g1,l -r %t-foo.bc,_g2,l \
; RUN:    -r %t-goo.bc,_g,l -r %t-goo.bc,_g1,l -r %t-goo.bc,_g2,l
; RUN: llvm-nm %tout-nowrite.1 | FileCheck %s --check-prefix=NOWRITE
; RUN: llvm-nm %tout-nowrite.2 | FileCheck %s --check-prefix=NOWRITE

; No merge instance is locally created as each module has a singltone function.
; NOWRITE-NOT: _f1.Tgm
; NOWRITE-NOT: _f2.Tgm

; RUN: llvm-lto2 run -enable-global-merge-func=true -codegen-data-generate=true %t-foo.bc %t-goo.bc -o %tout-nowrite \
; RUN:    -r %t-foo.bc,_f1,px \
; RUN:    -r %t-goo.bc,_f2,px \
; RUN:    -r %t-foo.bc,_g,l -r %t-foo.bc,_g1,l -r %t-foo.bc,_g2,l \
; RUN:    -r %t-goo.bc,_g,l -r %t-goo.bc,_g1,l -r %t-goo.bc,_g2,l
; RUN: llvm-nm %tout-nowrite.1 | FileCheck %s --check-prefix=WRITE
; RUN: llvm-nm %tout-nowrite.2 | FileCheck %s --check-prefix=WRITE
; RUN: llvm-objdump -h %tout-nowrite.1 | FileCheck %s --check-prefix=SECTNAME
; RUN: llvm-objdump -h %tout-nowrite.2 | FileCheck %s --check-prefix=SECTNAME

; On a write mode, no merging happens yet for each module.
; We only create stable functions and publish them into __llvm_merge section for each object.
; WRITE-NOT: _f1.Tgm
; WRITE-NOT: _f2.Tgm
; SECTNAME: __llvm_merge

; Merge the cgdata using llvm-cgdata.
; We now validate the content of the merged cgdata.
; Two functions have the same hash with only one different constnat at a same location.
; RUN: llvm-cgdata --merge -o %tout.cgdata %tout-nowrite.1 %tout-nowrite.2
; RUN: llvm-cgdata --convert %tout.cgdata   -o - | FileCheck %s

; CHECK:      - Hash: [[#%d,HASH:]]
; CHECK-NEXT:   FunctionName: f1
; CHECK-NEXT:   ModuleName: {{.*}}
; CHECK-NEXT:   InstCount: [[#%d,INSTCOUNT:]]
; CHECK-NEXT:   IndexOperandHashes:
; CHECK-NEXT:     - InstIndex: [[#%d,INSTINDEX:]]
; CHECK-NEXT:       OpndIndex: [[#%d,OPNDINDEX:]]
; CHECK-NEXT:       OpndHash: {{.*}}

; CHECK:      - Hash: [[#%d,HASH]]
; CHECK-NEXT:   FunctionName: f2
; CHECK-NEXT:   ModuleName: {{.*}}
; CHECK-NEXT:   InstCount: [[#%d,INSTCOUNT]]
; CHECK-NEXT:   IndexOperandHashes:
; CHECK-NEXT:     - InstIndex: [[#%d,INSTINDEX]]
; CHECK-NEXT:       OpndIndex: [[#%d,OPNDINDEX]]
; CHECK-NEXT:       OpndHash: {{.*}}

;--- foo.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-unknown-ios12.0.0"

@g = external local_unnamed_addr global [0 x i32], align 4
@g1 = external global i32, align 4
@g2 = external global i32, align 4

define i32 @f1(i32 %a) {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], [0 x i32]* @g, i64 0, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %1 = load volatile i32, i32* @g1, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 1
  ret i32 %add
}

;--- goo.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-unknown-ios12.0.0"

@g = external local_unnamed_addr global [0 x i32], align 4
@g1 = external global i32, align 4
@g2 = external global i32, align 4

define i32 @f2(i32 %a) {
entry:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds [0 x i32], [0 x i32]* @g, i64 0, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %1 = load volatile i32, i32* @g2, align 4
  %mul = mul nsw i32 %1, %0
  %add = add nsw i32 %mul, 1
  ret i32 %add
}
