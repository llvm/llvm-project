; RUN: llc < %s -mtriple=i686-apple-darwin9 -mattr=sse4a | FileCheck %s

define void @test1(i8* %p, <4 x float> %a) nounwind optsize ssp {
; CHECK: test1:
; CHECK: movntss
  tail call void @llvm.x86.sse4a.movnt.ss(i8* %p, <4 x float> %a) nounwind
  ret void
}

declare void @llvm.x86.sse4a.movnt.ss(i8*, <4 x float>)

define void @test2(i8* %p, <2 x double> %a) nounwind optsize ssp {
; CHECK: test2:
; CHECK: movntsd
  tail call void @llvm.x86.sse4a.movnt.sd(i8* %p, <2 x double> %a) nounwind
  ret void
}

declare void @llvm.x86.sse4a.movnt.sd(i8*, <2 x double>)

define <2 x i64> @test3(<2 x i64> %x) nounwind uwtable ssp {
; CHECK: test3:
; CHECK: extrq
  %1 = tail call <2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64> %x, i8 3, i8 2)
  ret <2 x i64> %1
}

declare <2 x i64> @llvm.x86.sse4a.extrqi(<2 x i64>, i8, i8) nounwind

define <2 x i64> @test4(<2 x i64> %x, <2 x i64> %y) nounwind uwtable ssp {
; CHECK: test4:
; CHECK: extrq
  %1 = bitcast <2 x i64> %y to <16 x i8>
  %2 = tail call <2 x i64> @llvm.x86.sse4a.extrq(<2 x i64> %x, <16 x i8> %1) nounwind
  ret <2 x i64> %2
}

declare <2 x i64> @llvm.x86.sse4a.extrq(<2 x i64>, <16 x i8>) nounwind

define <2 x i64> @test5(<2 x i64> %x, <2 x i64> %y) nounwind uwtable ssp {
; CHECK: test5:
; CHECK: insertq
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64> %x, <2 x i64> %y, i8 5, i8 6)
  ret <2 x i64> %1
}

declare <2 x i64> @llvm.x86.sse4a.insertqi(<2 x i64>, <2 x i64>, i8, i8) nounwind

define <2 x i64> @test6(<2 x i64> %x, <2 x i64> %y) nounwind uwtable ssp {
; CHECK: test6:
; CHECK: insertq
  %1 = tail call <2 x i64> @llvm.x86.sse4a.insertq(<2 x i64> %x, <2 x i64> %y) nounwind
  ret <2 x i64> %1
}

declare <2 x i64> @llvm.x86.sse4a.insertq(<2 x i64>, <2 x i64>) nounwind
