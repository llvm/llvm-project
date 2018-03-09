; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 -mattr=+ptx60 | FileCheck %s

declare i1 @llvm.nvvm.vote.all(i1)
; CHECK-LABEL: .func{{.*}}vote.all
define i1 @vote.all(i1 %pred) {
  ; CHECK: vote.all.pred
  %val = call i1 @llvm.nvvm.vote.all(i1 %pred)
  ret i1 %val
}

declare i1 @llvm.nvvm.vote.any(i1)
; CHECK-LABEL: .func{{.*}}vote.any
define i1 @vote.any(i1 %pred) {
  ; CHECK: vote.any.pred
  %val = call i1 @llvm.nvvm.vote.any(i1 %pred)
  ret i1 %val
}

declare i1 @llvm.nvvm.vote.uni(i1)
; CHECK-LABEL: .func{{.*}}vote.uni
define i1 @vote.uni(i1 %pred) {
  ; CHECK: vote.uni.pred
  %val = call i1 @llvm.nvvm.vote.uni(i1 %pred)
  ret i1 %val
}

declare i32 @llvm.nvvm.vote.ballot(i1)
; CHECK-LABEL: .func{{.*}}vote.ballot
define i32 @vote.ballot(i1 %pred) {
  ; CHECK: vote.ballot.b32
  %val = call i32 @llvm.nvvm.vote.ballot(i1 %pred)
  ret i32 %val
}

declare i1 @llvm.nvvm.vote.all.sync(i32, i1)
; CHECK-LABEL: .func{{.*}}vote.sync.all
define i1 @vote.sync.all(i32 %mask, i1 %pred) {
  ; CHECK: vote.sync.all.pred
  %val = call i1 @llvm.nvvm.vote.all.sync(i32 %mask, i1 %pred)
  ret i1 %val
}

declare i1 @llvm.nvvm.vote.any.sync(i32, i1)
; CHECK-LABEL: .func{{.*}}vote.sync.any
define i1 @vote.sync.any(i32 %mask, i1 %pred) {
  ; CHECK: vote.sync.any.pred
  %val = call i1 @llvm.nvvm.vote.any.sync(i32 %mask, i1 %pred)
  ret i1 %val
}

declare i1 @llvm.nvvm.vote.uni.sync(i32, i1)
; CHECK-LABEL: .func{{.*}}vote.sync.uni
define i1 @vote.sync.uni(i32 %mask, i1 %pred) {
  ; CHECK: vote.sync.uni.pred
  %val = call i1 @llvm.nvvm.vote.uni.sync(i32 %mask, i1 %pred)
  ret i1 %val
}

declare i32 @llvm.nvvm.vote.ballot.sync(i32, i1)
; CHECK-LABEL: .func{{.*}}vote.sync.ballot
define i32 @vote.sync.ballot(i32 %mask, i1 %pred) {
  ; CHECK: vote.sync.ballot.b32
  %val = call i32 @llvm.nvvm.vote.ballot.sync(i32 %mask, i1 %pred)
  ret i32 %val
}
