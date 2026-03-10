; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_30 -mattr=+ptx60 | FileCheck %s
; RUN: %if ptxas-isa-6.0 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_30 -mattr=+ptx60 | %ptxas-verify %}

declare i1 @llvm.nvvm.vote.all(i1)
; CHECK-LABEL: .func{{.*}}vote_all
define i1 @vote_all(i1 %pred) {
  ; CHECK: vote.all.pred
  %val = call i1 @llvm.nvvm.vote.all(i1 %pred)
  ret i1 %val
}

declare i1 @llvm.nvvm.vote.any(i1)
; CHECK-LABEL: .func{{.*}}vote_any
define i1 @vote_any(i1 %pred) {
  ; CHECK: vote.any.pred
  %val = call i1 @llvm.nvvm.vote.any(i1 %pred)
  ret i1 %val
}

declare i1 @llvm.nvvm.vote.uni(i1)
; CHECK-LABEL: .func{{.*}}vote_uni
define i1 @vote_uni(i1 %pred) {
  ; CHECK: vote.uni.pred
  %val = call i1 @llvm.nvvm.vote.uni(i1 %pred)
  ret i1 %val
}

declare i32 @llvm.nvvm.vote.ballot(i1)
; CHECK-LABEL: .func{{.*}}vote_ballot
define i32 @vote_ballot(i1 %pred) {
  ; CHECK: vote.ballot.b32
  %val = call i32 @llvm.nvvm.vote.ballot(i1 %pred)
  ret i32 %val
}

declare i1 @llvm.nvvm.vote.all.sync(i32, i1)
; CHECK-LABEL: .func{{.*}}vote_sync_all
define i1 @vote_sync_all(i32 %mask, i1 %pred) {
  ; CHECK: vote.sync.all.pred
  %val = call i1 @llvm.nvvm.vote.all.sync(i32 %mask, i1 %pred)
  ret i1 %val
}

declare i1 @llvm.nvvm.vote.any.sync(i32, i1)
; CHECK-LABEL: .func{{.*}}vote_sync_any
define i1 @vote_sync_any(i32 %mask, i1 %pred) {
  ; CHECK: vote.sync.any.pred
  %val = call i1 @llvm.nvvm.vote.any.sync(i32 %mask, i1 %pred)
  ret i1 %val
}

declare i1 @llvm.nvvm.vote.uni.sync(i32, i1)
; CHECK-LABEL: .func{{.*}}vote_sync_uni
define i1 @vote_sync_uni(i32 %mask, i1 %pred) {
  ; CHECK: vote.sync.uni.pred
  %val = call i1 @llvm.nvvm.vote.uni.sync(i32 %mask, i1 %pred)
  ret i1 %val
}

declare i32 @llvm.nvvm.vote.ballot.sync(i32, i1)
; CHECK-LABEL: .func{{.*}}vote_sync_ballot
define i32 @vote_sync_ballot(i32 %mask, i1 %pred) {
  ; CHECK: vote.sync.ballot.b32
  %val = call i32 @llvm.nvvm.vote.ballot.sync(i32 %mask, i1 %pred)
  ret i32 %val
}
