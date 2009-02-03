; RUN: llvm-as < %s | llc -march=x86 -mtriple=i686-pc-linux-gnu | grep movs | count 1

@A = global [32 x i32] zeroinitializer
@B = global [32 x i32] zeroinitializer

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)

define void @main() nounwind {
  ; dword copy
  call void @llvm.memcpy.i32(i8* bitcast ([32 x i32]* @A to i8*),
                           i8* bitcast ([32 x i32]* @B to i8*),
                           i32 128, i32 4 )

  ; word copy
  call void @llvm.memcpy.i32( i8* bitcast ([32 x i32]* @A to i8*),
                           i8* bitcast ([32 x i32]* @B to i8*),
                           i32 128, i32 2 )

  ; byte copy
  call void @llvm.memcpy.i32( i8* bitcast ([32 x i32]* @A to i8*),
                           i8* bitcast ([32 x i32]* @B to i8*),
                            i32 128, i32 1 )

  ret void
}
