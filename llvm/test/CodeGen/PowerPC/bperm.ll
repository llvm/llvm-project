; RUN: llc -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define zeroext i32 @bs4(i32 zeroext %a) #0 {
entry:
  %0 = tail call i32 @llvm.bswap.i32(i32 %a)
  ret i32 %0

; CHECK-LABEL: @bs4
; CHECK: rlwinm [[REG1:[0-9]+]], 3, 8, 0, 31
; CHECK: rlwimi [[REG1]], 3, 24, 16, 23
; CHECK: rlwimi [[REG1]], 3, 24, 0, 7
; CHECK: mr 3, [[REG1]]
; CHECK: blr
}

define i64 @bs8(i64 %x) #0 {
entry:
  %0 = tail call i64 @llvm.bswap.i64(i64 %x)
  ret i64 %0

; CHECK-LABEL: @bs8
; CHECK-DAG: rotldi [[REG1:[0-9]+]], 3, 16
; CHECK-DAG: rotldi [[REG2:[0-9]+]], 3, 8
; CHECK-DAG: rotldi [[REG3:[0-9]+]], 3, 24
; CHECK-DAG: rldimi [[REG2]], [[REG1]], 8, 48
; CHECK-DAG: rotldi [[REG4:[0-9]+]], 3, 32
; CHECK-DAG: rldimi [[REG2]], [[REG3]], 16, 40
; CHECK-DAG: rotldi [[REG5:[0-9]+]], 3, 48
; CHECK-DAG: rldimi [[REG2]], [[REG4]], 24, 32
; CHECK-DAG: rotldi [[REG6:[0-9]+]], 3, 56
; CHECK-DAG: rldimi [[REG2]], [[REG5]], 40, 16
; CHECK-DAG: rldimi [[REG2]], [[REG6]], 48, 8
; CHECK-DAG: rldimi [[REG2]], 3, 56, 0
; CHECK: mr 3, [[REG2]]
; CHECK: blr
}

define i64 @test1(i64 %i0, i64 %i1) #0 {
entry:
  %0 = lshr i64 %i1, 8
  %and = and i64 %0, 5963776000
  ret i64 %and

; CHECK-LABEL: @test1
; CHECK-DAG: li [[REG1:[0-9]+]], 11375
; CHECK-DAG: rotldi [[REG3:[0-9]+]], 4, 56
; CHECK-DAG: sldi [[REG2:[0-9]+]], [[REG1]], 19
; CHECK: and 3, [[REG3]], [[REG2]]
; CHECK: blr
}

define i64 @test2(i64 %i0, i64 %i1) #0 {
entry:
  %0 = lshr i64 %i1, 6
  %and = and i64 %0, 133434808670355456
  ret i64 %and

; CHECK-LABEL: @test2
; CHECK-DAG: lis [[REG1:[0-9]+]], 474
; CHECK-DAG: rotldi [[REG5:[0-9]+]], 4, 58
; CHECK-DAG: ori [[REG2:[0-9]+]], [[REG1]], 3648
; CHECK-DAG: sldi [[REG3:[0-9]+]], [[REG2]], 32
; CHECK-DAG: oris [[REG4:[0-9]+]], [[REG3]], 25464
; CHECK: and 3, [[REG5]], [[REG4]]
; CHECK: blr
}

define i64 @test3(i64 %i0, i64 %i1) #0 {
entry:
  %0 = shl i64 %i0, 34
  %and = and i64 %0, 191795733152661504
  ret i64 %and

; CHECK-LABEL: @test3
; CHECK-DAG: lis [[REG1:[0-9]+]], 170
; CHECK-DAG: rotldi [[REG4:[0-9]+]], 3, 34
; CHECK-DAG: ori [[REG2:[0-9]+]], [[REG1]], 22861
; CHECK-DAG: sldi [[REG3:[0-9]+]], [[REG2]], 34
; CHECK: and 3, [[REG4]], [[REG3]]
; CHECK: blr
}

define i64 @test4(i64 %i0, i64 %i1) #0 {
entry:
  %0 = lshr i64 %i1, 15
  %and = and i64 %0, 58195968
  ret i64 %and

; CHECK-LABEL: @test4
; CHECK: rotldi [[REG1:[0-9]+]], 4, 49
; CHECK: andis. 3, [[REG1]], 888
; CHECK: blr
}

define i64 @test5(i64 %i0, i64 %i1) #0 {
entry:
  %0 = shl i64 %i1, 12
  %and = and i64 %0, 127252959854592
  ret i64 %and

; CHECK-LABEL: @test5
; CHECK-DAG: lis [[REG1:[0-9]+]], 3703
; CHECK-DAG: rotldi [[REG4:[0-9]+]], 4, 12
; CHECK-DAG: ori [[REG2:[0-9]+]], [[REG1]], 35951
; CHECK-DAG: sldi [[REG3:[0-9]+]], [[REG2]], 19
; CHECK: and 3, [[REG4]], [[REG3]]
; CHECK: blr
}

; Function Attrs: nounwind readnone
define zeroext i32 @test6(i32 zeroext %x) #0 {
entry:
  %and = lshr i32 %x, 16
  %shr = and i32 %and, 255
  %and1 = shl i32 %x, 16
  %shl = and i32 %and1, 16711680
  %or = or i32 %shr, %shl
  ret i32 %or

; CHECK-LABEL: @test6
; CHECK: rlwinm [[REG1:[0-9]+]], 3, 16, 24, 31
; CHECK: rlwimi [[REG1]], 3, 16, 8, 15
; CHECK: mr 3, [[REG1]]
; CHECK: blr
}

define i64 @test7(i64 %i0, i64 %i1) #0 {
entry:
  %0 = lshr i64 %i0, 5
  %and = and i64 %0, 58195968
  ret i64 %and

; CHECK-LABEL: @test7
; CHECK: rlwinm [[REG1:[0-9]+]], 3, 27, 9, 12
; CHECK: rlwimi [[REG1]], 3, 27, 6, 7
; CHECK: mr 3, [[REG1]]
; CHECK: blr
}

define i64 @test8(i64 %i0, i64 %i1) #0 {
entry:
  %0 = lshr i64 %i0, 1
  %and = and i64 %0, 169172533248
  ret i64 %and

; CHECK-LABEL: @test8
; CHECK-DAG: lis [[REG1:[0-9]+]], 4
; CHECK-DAG: rotldi [[REG4:[0-9]+]], 3, 63
; CHECK-DAG: ori [[REG2:[0-9]+]], [[REG1]], 60527
; CHECK-DAG: sldi [[REG3:[0-9]+]], [[REG2]], 19
; CHECK: and 3, [[REG4]], [[REG3]]
; CHECK: blr
}

define i64 @test9(i64 %i0, i64 %i1) #0 {
entry:
  %0 = lshr i64 %i1, 14
  %and = and i64 %0, 18848677888
  %1 = shl i64 %i1, 51
  %and3 = and i64 %1, 405323966463344640
  %or4 = or i64 %and, %and3
  ret i64 %or4

; CHECK-LABEL: @test9
; CHECK-DAG: lis [[REG1:[0-9]+]], 1440
; CHECK-DAG: rotldi [[REG5:[0-9]+]], 4, 62
; CHECK-DAG: rotldi [[REG6:[0-9]+]], 4, 50
; CHECK-DAG: ori [[REG2:[0-9]+]], [[REG1]], 4
; CHECK-DAG: rldimi [[REG6]], [[REG5]], 53, 0
; CHECK-DAG: sldi [[REG3:[0-9]+]], [[REG2]], 32
; CHECK-DAG: oris [[REG4:[0-9]+]], [[REG3]], 25464
; CHECK: and 3, [[REG6]], [[REG4]]
; CHECK: blr
}

define i64 @test10(i64 %i0, i64 %i1) #0 {
entry:
  %0 = shl i64 %i0, 37
  %and = and i64 %0, 15881483390550016
  %1 = shl i64 %i0, 25
  %and3 = and i64 %1, 2473599172608
  %or4 = or i64 %and, %and3
  ret i64 %or4

; CHECK-LABEL: @test10
; CHECK-DAG: lis [[REG1:[0-9]+]], 1
; CHECK-DAG: rotldi [[REG6:[0-9]+]], 3, 25
; CHECK-DAG: rotldi [[REG7:[0-9]+]], 3, 37
; CHECK-DAG: ori [[REG2:[0-9]+]], [[REG1]], 8183
; CHECK-DAG: ori [[REG3:[0-9]+]], [[REG1]], 50017
; CHECK-DAG: sldi [[REG4:[0-9]+]], [[REG2]], 25
; CHECK-DAG: sldi [[REG5:[0-9]+]], [[REG3]], 37
; CHECK-DAG: and [[REG8:[0-9]+]], [[REG6]], [[REG4]]
; CHECK-DAG: and [[REG9:[0-9]+]], [[REG7]], [[REG5]]
; CHECK: or 3, [[REG9]], [[REG8]]
; CHECK: blr
}

define i64 @test11(i64 %x) #0 {
entry:
  %and = and i64 %x, 4294967295
  %shl = shl i64 %x, 32
  %or = or i64 %and, %shl
  ret i64 %or

; CHECK-LABEL: @test11
; CHECK: rlwinm 3, 3, 0, 1, 0
; CHECK: blr
}

define i64 @test12(i64 %x) #0 {
entry:
  %and = and i64 %x, 4294905855
  %shl = shl i64 %x, 32
  %or = or i64 %and, %shl
  ret i64 %or

; CHECK-LABEL: @test12
; CHECK: rlwinm 3, 3, 0, 20, 15
; CHECK: blr
}

define i64 @test13(i64 %x) #0 {
entry:
  %shl = shl i64 %x, 4
  %and = and i64 %shl, 240
  %shr = lshr i64 %x, 28
  %and1 = and i64 %shr, 15
  %or = or i64 %and, %and1
  ret i64 %or

; CHECK-LABEL: @test13
; CHECK: rlwinm 3, 3, 4, 24, 31
; CHECK: blr
}

define i64 @test14(i64 %x) #0 {
entry:
  %shl = shl i64 %x, 4
  %and = and i64 %shl, 240
  %shr = lshr i64 %x, 28
  %and1 = and i64 %shr, 15
  %and2 = and i64 %x, -4294967296
  %or = or i64 %and1, %and2
  %or3 = or i64 %or, %and
  ret i64 %or3

; CHECK-LABEL: @test14
; CHECK: rldicr [[REG1:[0-9]+]], 3, 0, 31
; CHECK: rlwimi [[REG1]], 3, 4, 24, 31
; CHECK: mr 3, [[REG1]]
; CHECK: blr
}

define i64 @test15(i64 %x) #0 {
entry:
  %shl = shl i64 %x, 4
  %and = and i64 %shl, 240
  %shr = lshr i64 %x, 28
  %and1 = and i64 %shr, 15
  %and2 = and i64 %x, -256
  %or = or i64 %and1, %and2
  %or3 = or i64 %or, %and
  ret i64 %or3

; CHECK-LABEL: @test15
; CHECK: rlwimi 3, 3, 4, 24, 31
; CHECK: blr
}

; Function Attrs: nounwind readnone
declare i32 @llvm.bswap.i32(i32) #0
declare i64 @llvm.bswap.i64(i64) #0

attributes #0 = { nounwind readnone }

