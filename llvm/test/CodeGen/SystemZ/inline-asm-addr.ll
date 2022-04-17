; RUN: llc -mtriple=s390x-linux-gnu < %s | FileCheck %s

@Addr = global i64 0, align 8
@A = global i64* null, align 8
@Idx = global i64 0, align 8

define i64 @fun_BD12_p() {
; CHECK-LABEL: fun_BD12_p:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r1)
entry:
  %0 = load i64*, i64** @A
  %arrayidx = getelementptr inbounds i64, i64* %0, i64 100
  %1 = tail call i64 asm "lay $0, $1", "=r,p"(i64* nonnull %arrayidx)
  store i64 %1, i64* @Addr
  ret i64 %1
}

define i64 @fun_BDX12_p() {
; CHECK-LABEL: fun_BDX12_p:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r1,%r2)
entry:
  %0 = load i64*, i64** @A
  %1 = load i64, i64* @Idx
  %add = add nsw i64 %1, 100
  %arrayidx = getelementptr inbounds i64, i64* %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,p"(i64* %arrayidx)
  store i64 %2, i64* @Addr
  ret i64 %2
}

define i64 @fun_BD20_p() {
; CHECK-LABEL: fun_BD20_p:
; CHECK: #APP
; CHECK: lay	%r2, 8000(%r1)
entry:
  %0 = load i64*, i64** @A
  %arrayidx = getelementptr inbounds i64, i64* %0, i64 1000
  %1 = tail call i64 asm "lay $0, $1", "=r,p"(i64* nonnull %arrayidx)
  store i64 %1, i64* @Addr
  ret i64 %1
}

define i64 @fun_BDX20_p() {
; CHECK-LABEL: fun_BDX20_p:
; CHECK: #APP
; CHECK: lay	%r2, 8000(%r1,%r2)
entry:
  %0 = load i64*, i64** @A
  %1 = load i64, i64* @Idx
  %add = add nsw i64 %1, 1000
  %arrayidx = getelementptr inbounds i64, i64* %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,p"(i64* %arrayidx)
  store i64 %2, i64* @Addr
  ret i64 %2
}
