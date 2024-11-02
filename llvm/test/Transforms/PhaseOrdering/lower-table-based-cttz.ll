;; This tests lowering of the implementations of table-based ctz
;; algorithm to the llvm.cttz instruction in the -O3 case.

;; C producer:
;; int ctz1 (unsigned x)
;; {
;; static const char table[32] =
;;    {
;;     0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
;;      31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
;;    };
;;   return table[((unsigned)((x & -x) * 0x077CB531U)) >> 27];
;; }
;; Compiled as: clang -O3 test.c -S -emit-llvm -Xclang -disable-llvm-optzns

; RUN: opt -O3 -S < %s | FileCheck %s

; CHECK: call i32 @llvm.cttz.i32

@ctz1.table = internal constant [32 x i8] c"\00\01\1C\02\1D\0E\18\03\1E\16\14\0F\19\11\04\08\1F\1B\0D\17\15\13\10\07\1A\0C\12\06\0B\05\0A\09", align 16

define i32 @ctz1(i32 noundef %x) {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %1 = load i32, ptr %x.addr, align 4
  %sub = sub i32 0, %1
  %and = and i32 %0, %sub
  %mul = mul i32 %and, 125613361
  %shr = lshr i32 %mul, 27
  %idxprom = zext i32 %shr to i64
  %arrayidx = getelementptr inbounds [32 x i8], ptr @ctz1.table, i64 0, i64 %idxprom
  %2 = load i8, ptr %arrayidx, align 1
  %conv = sext i8 %2 to i32
  ret i32 %conv
}
