; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s
; RUN: llvm-lto -thinlto -o %t2 %t.o
; RUN: llvm-bcanalyzer -dump %t2.thinlto.bc | FileCheck --check-prefix=COMBINED %s

; COMBINED: <TYPE_TESTS op0=-2012135647395072713/>
; COMBINED: <TYPE_TESTS op0=6699318081062747564 op1=-2012135647395072713/>
; COMBINED: <TYPE_TESTS op0=6699318081062747564/>

; CHECK: <TYPE_TESTS op0=6699318081062747564/>
define i1 @f() {
  %p = call i1 @llvm.type.test(i8* null, metadata !"foo")
  ret i1 %p
}

; CHECK: <TYPE_TESTS op0=6699318081062747564 op1=-2012135647395072713/>
define i1 @g() {
  %p = call i1 @llvm.type.test(i8* null, metadata !"foo")
  %q = call i1 @llvm.type.test(i8* null, metadata !"bar")
  %pq = and i1 %p, %q
  ret i1 %pq
}

; CHECK: <TYPE_TESTS op0=-2012135647395072713/>
define i1 @h() {
  %p = call i1 @llvm.type.test(i8* null, metadata !"bar")
  ret i1 %p
}

declare i1 @llvm.type.test(i8*, metadata) nounwind readnone
