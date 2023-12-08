; RUN: llc -mtriple=s390x-linux-gnu < %s | FileCheck %s

@Addr = global i64 0, align 8
@A = global ptr null, align 8
@Idx = global i64 0, align 8

define i64 @fun_BD12_Q() {
; CHECK-LABEL: fun_BD12_Q:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r1)
entry:
  %0 = load ptr, ptr @A
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 100
  %1 = tail call i64 asm "lay $0, $1", "=r,^ZQ"(ptr nonnull %arrayidx)
  store i64 %1, ptr @Addr
  ret i64 %1
}

define i64 @fun_BD12_R() {
; CHECK-LABEL: fun_BD12_R:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r1)
entry:
  %0 = load ptr, ptr @A
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 100
  %1 = tail call i64 asm "lay $0, $1", "=r,^ZR"(ptr nonnull %arrayidx)
  store i64 %1, ptr @Addr
  ret i64 %1
}

define i64 @fun_BD12_S() {
; CHECK-LABEL: fun_BD12_S:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r1)
entry:
  %0 = load ptr, ptr @A
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 100
  %1 = tail call i64 asm "lay $0, $1", "=r,^ZS"(ptr nonnull %arrayidx)
  store i64 %1, ptr @Addr
  ret i64 %1
}

define i64 @fun_BD12_T() {
; CHECK-LABEL: fun_BD12_T:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r1)
entry:
  %0 = load ptr, ptr @A
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 100
  %1 = tail call i64 asm "lay $0, $1", "=r,^ZT"(ptr nonnull %arrayidx)
  store i64 %1, ptr @Addr
  ret i64 %1
}

define i64 @fun_BD12_p() {
; CHECK-LABEL: fun_BD12_p:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r1)
entry:
  %0 = load ptr, ptr @A
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 100
  %1 = tail call i64 asm "lay $0, $1", "=r,p"(ptr nonnull %arrayidx)
  store i64 %1, ptr @Addr
  ret i64 %1
}

define i64 @fun_BDX12_Q() {
; CHECK-LABEL: fun_BDX12_Q:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r2)
entry:
  %0 = load ptr, ptr @A
  %1 = load i64, ptr @Idx
  %add = add nsw i64 %1, 100
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,^ZQ"(ptr %arrayidx)
  store i64 %2, ptr @Addr
  ret i64 %2
}

define i64 @fun_BDX12_R() {
; CHECK-LABEL: fun_BDX12_R:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r1,%r2)
entry:
  %0 = load ptr, ptr @A
  %1 = load i64, ptr @Idx
  %add = add nsw i64 %1, 100
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,^ZR"(ptr %arrayidx)
  store i64 %2, ptr @Addr
  ret i64 %2
}

define i64 @fun_BDX12_S() {
; CHECK-LABEL: fun_BDX12_S:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r2)
entry:
  %0 = load ptr, ptr @A
  %1 = load i64, ptr @Idx
  %add = add nsw i64 %1, 100
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,^ZS"(ptr %arrayidx)
  store i64 %2, ptr @Addr
  ret i64 %2
}

define i64 @fun_BDX12_T() {
; CHECK-LABEL: fun_BDX12_T:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r1,%r2)
entry:
  %0 = load ptr, ptr @A
  %1 = load i64, ptr @Idx
  %add = add nsw i64 %1, 100
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,^ZT"(ptr %arrayidx)
  store i64 %2, ptr @Addr
  ret i64 %2
}

define i64 @fun_BDX12_p() {
; CHECK-LABEL: fun_BDX12_p:
; CHECK: #APP
; CHECK: lay	%r2, 800(%r1,%r2)
entry:
  %0 = load ptr, ptr @A
  %1 = load i64, ptr @Idx
  %add = add nsw i64 %1, 100
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,p"(ptr %arrayidx)
  store i64 %2, ptr @Addr
  ret i64 %2
}

define i64 @fun_BD20_Q() {
; CHECK-LABEL: fun_BD20_Q:
; CHECK: #APP
; CHECK: lay	%r2, 0(%r2)
entry:
  %0 = load ptr, ptr @A
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 1000
  %1 = tail call i64 asm "lay $0, $1", "=r,^ZQ"(ptr nonnull %arrayidx)
  store i64 %1, ptr @Addr
  ret i64 %1
}

define i64 @fun_BD20_R() {
; CHECK-LABEL: fun_BD20_R:
; CHECK: #APP
; CHECK: lay	%r2, 0(%r2)
entry:
  %0 = load ptr, ptr @A
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 1000
  %1 = tail call i64 asm "lay $0, $1", "=r,^ZR"(ptr nonnull %arrayidx)
  store i64 %1, ptr @Addr
  ret i64 %1
}

define i64 @fun_BD20_S() {
; CHECK-LABEL: fun_BD20_S:
; CHECK: #APP
; CHECK: lay	%r2, 8000(%r1)
entry:
  %0 = load ptr, ptr @A
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 1000
  %1 = tail call i64 asm "lay $0, $1", "=r,^ZS"(ptr nonnull %arrayidx)
  store i64 %1, ptr @Addr
  ret i64 %1
}

define i64 @fun_BD20_T() {
; CHECK-LABEL: fun_BD20_T:
; CHECK: #APP
; CHECK: lay	%r2, 8000(%r1)
entry:
  %0 = load ptr, ptr @A
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 1000
  %1 = tail call i64 asm "lay $0, $1", "=r,^ZT"(ptr nonnull %arrayidx)
  store i64 %1, ptr @Addr
  ret i64 %1
}

define i64 @fun_BD20_p() {
; CHECK-LABEL: fun_BD20_p:
; CHECK: #APP
; CHECK: lay	%r2, 8000(%r1)
entry:
  %0 = load ptr, ptr @A
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 1000
  %1 = tail call i64 asm "lay $0, $1", "=r,p"(ptr nonnull %arrayidx)
  store i64 %1, ptr @Addr
  ret i64 %1
}

define i64 @fun_BDX20_Q() {
; CHECK-LABEL: fun_BDX20_Q:
; CHECK: #APP
; CHECK: lay	%r2, 0(%r1)
entry:
  %0 = load ptr, ptr @A
  %1 = load i64, ptr @Idx
  %add = add nsw i64 %1, 1000
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,^ZQ"(ptr %arrayidx)
  store i64 %2, ptr @Addr
  ret i64 %2
}

define i64 @fun_BDX20_R() {
; CHECK-LABEL: fun_BDX20_R:
; CHECK: #APP
; CHECK: lay	%r2, 0(%r1)
entry:
  %0 = load ptr, ptr @A
  %1 = load i64, ptr @Idx
  %add = add nsw i64 %1, 1000
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,^ZR"(ptr %arrayidx)
  store i64 %2, ptr @Addr
  ret i64 %2
}

define i64 @fun_BDX20_S() {
; CHECK-LABEL: fun_BDX20_S:
; CHECK: #APP
; CHECK: lay	%r2, 8000(%r2)
entry:
  %0 = load ptr, ptr @A
  %1 = load i64, ptr @Idx
  %add = add nsw i64 %1, 1000
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,^ZS"(ptr %arrayidx)
  store i64 %2, ptr @Addr
  ret i64 %2
}

define i64 @fun_BDX20_T() {
; CHECK-LABEL: fun_BDX20_T:
; CHECK: #APP
; CHECK: lay	%r2, 8000(%r1,%r2)
entry:
  %0 = load ptr, ptr @A
  %1 = load i64, ptr @Idx
  %add = add nsw i64 %1, 1000
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,^ZT"(ptr %arrayidx)
  store i64 %2, ptr @Addr
  ret i64 %2
}

define i64 @fun_BDX20_p() {
; CHECK-LABEL: fun_BDX20_p:
; CHECK: #APP
; CHECK: lay	%r2, 8000(%r1,%r2)
entry:
  %0 = load ptr, ptr @A
  %1 = load i64, ptr @Idx
  %add = add nsw i64 %1, 1000
  %arrayidx = getelementptr inbounds i64, ptr %0, i64 %add
  %2 = tail call i64 asm "lay $0, $1", "=r,p"(ptr %arrayidx)
  store i64 %2, ptr @Addr
  ret i64 %2
}
