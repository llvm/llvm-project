; RUN: llc -mtriple=mipsel -relocation-model=pic -O0 -fast-isel=true -mcpu=mips32r2 \
; RUN:     < %s -verify-machineinstrs | FileCheck %s

define void @testeq(i32, i32) {
; CHECK-LABEL: testeq:
; CHECK:       andi $[[REG0:[0-9]+]], $4, 1
; CHECK:       andi $[[REG1:[0-9]+]], $5, 1
; CHECK:       beq $[[REG0]], $[[REG1]],
  %3 = trunc i32 %0 to i1
  %4 = trunc i32 %1 to i1
  %5 = icmp eq i1 %3, %4
  br i1 %5, label %end, label %trap
trap:
  call void @llvm.trap()
  br label %end
end:
  ret void
}


define void @testne(i32, i32) {
; CHECK-LABEL: testne:
; CHECK:       andi $[[REG0:[0-9]+]], $4, 1
; CHECK:       andi $[[REG1:[0-9]+]], $5, 1
; CHECK:       bne $[[REG0]], $[[REG1]],
  %3 = trunc i32 %0 to i1
  %4 = trunc i32 %1 to i1
  %5 = icmp ne i1 %3, %4
  br i1 %5, label %end, label %trap
trap:
  call void @llvm.trap()
  br label %end
end:
  ret void
}


define void @testugt(i32, i32) {
; CHECK-LABEL: testugt:
; CHECK:       andi $[[REG0:[0-9]+]], $4, 1
; CHECK:       andi $[[REG1:[0-9]+]], $5, 1
; CHECK:       sltu $[[REG2:[0-9]+]], $[[REG1]], $[[REG0]]
; CHECK:       bnez $[[REG2]],
  %3 = trunc i32 %0 to i1
  %4 = trunc i32 %1 to i1
  %5 = icmp ugt i1 %3, %4
  br i1 %5, label %end, label %trap
trap:
  call void @llvm.trap()
  br label %end
end:
  ret void
}


define void @testuge(i32, i32) {
; CHECK-LABEL: testuge:
; CHECK:       andi $[[REG0:[0-9]+]], $4, 1
; CHECK:       andi $[[REG1:[0-9]+]], $5, 1
; CHECK:       sltu $[[REG2:[0-9]+]], $[[REG0]], $[[REG1]]
; CHECK:       beqz $[[REG2]],
  %3 = trunc i32 %0 to i1
  %4 = trunc i32 %1 to i1
  %5 = icmp uge i1 %3, %4
  br i1 %5, label %end, label %trap
trap:
  call void @llvm.trap()
  br label %end
end:
  ret void
}


define void @testult(i32, i32) {
; CHECK-LABEL: testult:
; CHECK:       andi $[[REG0:[0-9]+]], $4, 1
; CHECK:       andi $[[REG1:[0-9]+]], $5, 1
; CHECK:       sltu $[[REG2:[0-9]+]], $[[REG0]], $[[REG1]]
; CHECK:       bnez $[[REG2]],
  %3 = trunc i32 %0 to i1
  %4 = trunc i32 %1 to i1
  %5 = icmp ult i1 %3, %4
  br i1 %5, label %end, label %trap
trap:
  call void @llvm.trap()
  br label %end
end:
  ret void
}


define void @testule(i32, i32) {
; CHECK:       andi $[[REG0:[0-9]+]], $4, 1
; CHECK:       andi $[[REG1:[0-9]+]], $5, 1
; CHECK:       sltu $[[REG2:[0-9]+]], $[[REG1]], $[[REG0]]
; CHECK:       beqz $[[REG2]],
  %3 = trunc i32 %0 to i1
  %4 = trunc i32 %1 to i1
  %5 = icmp ule i1 %3, %4
  br i1 %5, label %end, label %trap
trap:
  call void @llvm.trap()
  br label %end
end:
  ret void
}


define void @testsgt(i32, i32) {
; CHECK-LABEL: testsgt:
; CHECK:       andi $[[REG0:[0-9]+]], $4, 1
; CHECK:       negu $[[REG2:[0-9]+]], $[[REG0]]
; CHECK:       andi $[[REG1:[0-9]+]], $5, 1
; CHECK:       negu $[[REG3:[0-9]+]], $[[REG1]]
; CHECK:       slt $[[REG4:[0-9]+]], $[[REG3]], $[[REG2]]
; CHECK:       bnez $[[REG4]],
  %3 = trunc i32 %0 to i1
  %4 = trunc i32 %1 to i1
  %5 = icmp sgt i1 %3, %4
  br i1 %5, label %end, label %trap
trap:
  call void @llvm.trap()
  br label %end
end:
  ret void
}


define void @testsge(i32, i32) {
; CHECK-LABEL: testsge:
; CHECK:       andi $[[REG0:[0-9]+]], $4, 1
; CHECK:       negu $[[REG0]], $[[REG0]]
; CHECK:       andi $[[REG1:[0-9]+]], $5, 1
; CHECK:       negu $[[REG1]], $[[REG1]]
; CHECK:       slt $[[REG2:[0-9]+]], $[[REG0]], $[[REG1]]
; CHECK:       beqz $[[REG2]],
  %3 = trunc i32 %0 to i1
  %4 = trunc i32 %1 to i1
  %5 = icmp sge i1 %3, %4
  br i1 %5, label %end, label %trap
trap:
  call void @llvm.trap()
  br label %end
end:
  ret void
}


define void @testslt(i32, i32) {
; CHECK-LABEL: testslt:
; CHECK:       andi $[[REG0:[0-9]+]], $4, 1
; CHECK:       negu $[[REG0]], $[[REG0]]
; CHECK:       andi $[[REG1:[0-9]+]], $5, 1
; CHECK:       negu $[[REG1]], $[[REG1]]
; CHECK:       slt $[[REG2:[0-9]+]], $[[REG0]], $[[REG1]]
; CHECK:       bnez $[[REG2]],
  %3 = trunc i32 %0 to i1
  %4 = trunc i32 %1 to i1
  %5 = icmp slt i1 %3, %4
  br i1 %5, label %end, label %trap
trap:
  call void @llvm.trap()
  br label %end
end:
  ret void
}


define void @testsle(i32, i32) {
; CHECK-LABEL: testsle:
; CHECK:       andi $[[REG0:[0-9]+]], $4, 1
; CHECK:       negu $[[REG2:[0-9]+]], $[[REG0]]
; CHECK:       andi $[[REG1:[0-9]+]], $5, 1
; CHECK:       negu $[[REG3:[0-9]+]], $[[REG1]]
; CHECK:       slt $[[REG4:[0-9]+]], $[[REG3]], $[[REG2]]
; CHECK:       beqz $[[REG4]],
  %3 = trunc i32 %0 to i1
  %4 = trunc i32 %1 to i1
  %5 = icmp sle i1 %3, %4
  br i1 %5, label %end, label %trap
trap:
  call void @llvm.trap()
  br label %end
end:
  ret void
}


declare void @llvm.trap()
