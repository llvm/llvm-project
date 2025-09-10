; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SetjmpLongjmp/C/WhileLoop.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SetjmpLongjmp/C/WhileLoop.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__jmp_buf_tag = type { [22 x i64], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [16 x i64] }

@.str = private unnamed_addr constant [16 x i8] c"Inside foo: %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [25 x i8] c"Return from longjmp: %d\0A\00", align 1

; Function Attrs: noreturn nounwind uwtable
define dso_local void @foo(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %1)
  tail call void @longjmp(ptr noundef %0, i32 noundef %1) #6
  unreachable
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: noreturn nounwind
declare void @longjmp(ptr noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca [1 x %struct.__jmp_buf_tag], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %2 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %150, label %4

4:                                                ; preds = %0
  %5 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %6 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %150, label %8

8:                                                ; preds = %4
  %9 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %10 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %150, label %12

12:                                               ; preds = %8
  %13 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %14 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %150, label %16

16:                                               ; preds = %12
  %17 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %14)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %18 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %150, label %20

20:                                               ; preds = %16
  %21 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %18)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %22 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %150, label %24

24:                                               ; preds = %20
  %25 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %22)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %26 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %150, label %28

28:                                               ; preds = %24
  %29 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %26)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %30 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %150, label %32

32:                                               ; preds = %28
  %33 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %30)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %34 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %35 = icmp eq i32 %34, 0
  br i1 %35, label %150, label %36

36:                                               ; preds = %32
  %37 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %34)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %38 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %150, label %40

40:                                               ; preds = %36
  %41 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %38)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %42 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %43 = icmp eq i32 %42, 0
  br i1 %43, label %150, label %44

44:                                               ; preds = %40
  %45 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %42)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %46 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %47 = icmp eq i32 %46, 0
  br i1 %47, label %150, label %48

48:                                               ; preds = %44
  %49 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %46)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %50 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %51 = icmp eq i32 %50, 0
  br i1 %51, label %150, label %52

52:                                               ; preds = %48
  %53 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %50)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %54 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %55 = icmp eq i32 %54, 0
  br i1 %55, label %150, label %56

56:                                               ; preds = %52
  %57 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %54)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %58 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %59 = icmp eq i32 %58, 0
  br i1 %59, label %150, label %60

60:                                               ; preds = %56
  %61 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %58)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %62 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %63 = icmp eq i32 %62, 0
  br i1 %63, label %150, label %64

64:                                               ; preds = %60
  %65 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %62)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %66 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %67 = icmp eq i32 %66, 0
  br i1 %67, label %150, label %68

68:                                               ; preds = %64
  %69 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %66)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %70 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %71 = icmp eq i32 %70, 0
  br i1 %71, label %150, label %72

72:                                               ; preds = %68
  %73 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %70)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %74 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %75 = icmp eq i32 %74, 0
  br i1 %75, label %150, label %76

76:                                               ; preds = %72
  %77 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %74)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %78 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %79 = icmp eq i32 %78, 0
  br i1 %79, label %150, label %80

80:                                               ; preds = %76
  %81 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %78)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %82 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %83 = icmp eq i32 %82, 0
  br i1 %83, label %150, label %84

84:                                               ; preds = %80
  %85 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %82)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %86 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %87 = icmp eq i32 %86, 0
  br i1 %87, label %150, label %88

88:                                               ; preds = %84
  %89 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %86)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %90 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %91 = icmp eq i32 %90, 0
  br i1 %91, label %150, label %92

92:                                               ; preds = %88
  %93 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %90)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %94 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %95 = icmp eq i32 %94, 0
  br i1 %95, label %150, label %96

96:                                               ; preds = %92
  %97 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %94)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %98 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %99 = icmp eq i32 %98, 0
  br i1 %99, label %150, label %100

100:                                              ; preds = %96
  %101 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %98)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %102 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %103 = icmp eq i32 %102, 0
  br i1 %103, label %150, label %104

104:                                              ; preds = %100
  %105 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %102)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %106 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %107 = icmp eq i32 %106, 0
  br i1 %107, label %150, label %108

108:                                              ; preds = %104
  %109 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %106)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %110 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %111 = icmp eq i32 %110, 0
  br i1 %111, label %150, label %112

112:                                              ; preds = %108
  %113 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %110)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %114 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %115 = icmp eq i32 %114, 0
  br i1 %115, label %150, label %116

116:                                              ; preds = %112
  %117 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %114)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %118 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %119 = icmp eq i32 %118, 0
  br i1 %119, label %150, label %120

120:                                              ; preds = %116
  %121 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %118)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %122 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %123 = icmp eq i32 %122, 0
  br i1 %123, label %150, label %124

124:                                              ; preds = %120
  %125 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %122)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %126 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %127 = icmp eq i32 %126, 0
  br i1 %127, label %150, label %128

128:                                              ; preds = %124
  %129 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %126)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %130 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %131 = icmp eq i32 %130, 0
  br i1 %131, label %150, label %132

132:                                              ; preds = %128
  %133 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %130)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %134 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %135 = icmp eq i32 %134, 0
  br i1 %135, label %150, label %136

136:                                              ; preds = %132
  %137 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %134)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %138 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %139 = icmp eq i32 %138, 0
  br i1 %139, label %150, label %140

140:                                              ; preds = %136
  %141 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %138)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %142 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %143 = icmp eq i32 %142, 0
  br i1 %143, label %150, label %144

144:                                              ; preds = %140
  %145 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %142)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  %146 = call i32 @_setjmp(ptr noundef nonnull %1) #8
  %147 = icmp eq i32 %146, 0
  br i1 %147, label %150, label %148

148:                                              ; preds = %144
  %149 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %146)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  ret i32 0

150:                                              ; preds = %144, %140, %136, %132, %128, %124, %120, %116, %112, %108, %104, %100, %96, %92, %88, %84, %80, %76, %72, %68, %64, %60, %56, %52, %48, %44, %40, %36, %32, %28, %24, %20, %16, %12, %8, %4, %0
  %151 = phi i32 [ 36, %0 ], [ 35, %4 ], [ 34, %8 ], [ 33, %12 ], [ 32, %16 ], [ 31, %20 ], [ 30, %24 ], [ 29, %28 ], [ 28, %32 ], [ 27, %36 ], [ 26, %40 ], [ 25, %44 ], [ 24, %48 ], [ 23, %52 ], [ 22, %56 ], [ 21, %60 ], [ 20, %64 ], [ 19, %68 ], [ 18, %72 ], [ 17, %76 ], [ 16, %80 ], [ 15, %84 ], [ 14, %88 ], [ 13, %92 ], [ 12, %96 ], [ 11, %100 ], [ 10, %104 ], [ 9, %108 ], [ 8, %112 ], [ 7, %116 ], [ 6, %120 ], [ 5, %124 ], [ 4, %128 ], [ 3, %132 ], [ 2, %136 ], [ 1, %140 ], [ 0, %144 ]
  %152 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %151)
  call void @longjmp(ptr noundef nonnull %1, i32 noundef %151) #6
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: nounwind returns_twice
declare i32 @_setjmp(ptr noundef) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

attributes #0 = { noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nounwind returns_twice "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { noreturn nounwind }
attributes #7 = { nounwind }
attributes #8 = { nounwind returns_twice }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
