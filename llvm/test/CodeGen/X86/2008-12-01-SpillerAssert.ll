; RUN: llvm-as < %s | llc -mtriple=x86_64-unknown-linux-gnu
; PR3124

        %struct.cpuinfo_x86 = type { i8, i8, i8, i8, i32, i8, i8, i8, i32, i32, [9 x i32], [16 x i8], [64 x i8], i32, i32, i32, i64, %struct.cpumask_t, i16, i16, i16, i16, i16, i16, i16, i16, i32 }
        %struct.cpumask_t = type { [1 x i64] }
@.str10 = external constant [70 x i8]           ; <[70 x i8]*> [#uses=1]

declare i32 @printk(i8*, ...)

define void @display_cacheinfo(%struct.cpuinfo_x86* %c) nounwind section ".cpuinit.text" {
entry:
        %asmtmp = tail call { i32, i32, i32, i32 } asm "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 -2147483643, i32 0) nounwind          ; <{ i32, i32, i32, i32 }> [#uses=0]
        %0 = tail call i32 (i8*, ...)* @printk(i8* getelementptr ([70 x i8]* @.str10, i32 0, i64 0), i32 0, i32 0, i32 0, i32 0) nounwind           ; <i32> [#uses=0]
        unreachable
}
