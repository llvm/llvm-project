; RUN: opt < %s -disable-output -passes=indvars

; Do not timeout (and do not crash).
;
; This test case used to take around 10 minutes to run (well, that of course
; depends on which kind of build that is used and on which kind of server the
; test is executed). There is a less reduced version of this test case in
; PR51869 that takes much longer time to execute (I've not seen that one
; terminate within reasonable time). Maybe this test case is reduced a bit too
; much if being considered as a regression tests that would timeout without
; the fix. It can at least be used to show compile time explosion that
; happened when using isKnownPredicate inside ScalarEvolution::isImpliedCond.

@v_228 = external dso_local global i32, align 1

; Function Attrs: nounwind
define dso_local i16 @main(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9, ptr %10, i1 %11) #0 {
  br i1 %11, label %27, label %13

13:                                               ; preds = %12
  %14 = load i32, ptr @v_228, align 1
  %15 = trunc i32 %14 to i16
  %16 = mul i16 %15, 2
  %17 = sub i16 10, %16
  %18 = icmp ult i16 10, %16
  %19 = icmp ult i16 %17, 1
  %20 = or i1 %18, %19
  br i1 %20, label %139, label %21

21:                                               ; preds = %13
  %22 = add i16 %16, 1
  %23 = sub i16 10, %22
  %24 = icmp ult i16 10, %22
  %25 = icmp ult i16 %23, 1
  %26 = or i1 %24, %25
  br i1 %26, label %139, label %27

27:                                               ; preds = %21, %12
  %28 = load i16, ptr %1, align 1
  br label %29

29:                                               ; preds = %29, %27
  %30 = phi i16 [ %28, %27 ], [ %31, %29 ]
  %31 = add i16 %30, 1
  %32 = icmp slt i16 %31, 28
  br i1 %32, label %29, label %33

33:                                               ; preds = %29
  %34 = load i16, ptr %2, align 1
  br label %35

35:                                               ; preds = %43, %33
  %36 = phi i16 [ %34, %33 ], [ %44, %43 ]
  %37 = sext i16 %36 to i32
  %38 = mul i32 %37, 2
  %39 = sub i32 56, %38
  %40 = icmp ult i32 56, %38
  %41 = icmp ult i32 %39, 2
  %42 = or i1 %40, %41
  br i1 %42, label %139, label %43

43:                                               ; preds = %35
  %44 = add i16 %36, 1
  %45 = icmp slt i16 %44, 28
  br i1 %45, label %35, label %46

46:                                               ; preds = %43
  %47 = load i16, ptr %3, align 1
  br label %48

48:                                               ; preds = %55, %46
  %49 = phi i16 [ %47, %46 ], [ %56, %55 ]
  %50 = mul i16 %49, 4
  %51 = sub i16 28, %50
  %52 = icmp ult i16 28, %50
  %53 = icmp ult i16 %51, 4
  %54 = or i1 %52, %53
  br i1 %54, label %139, label %55

55:                                               ; preds = %48
  %56 = add i16 %49, 1
  %57 = icmp slt i16 %56, 7
  br i1 %57, label %48, label %58

58:                                               ; preds = %55
  %59 = load i16, ptr %4, align 1
  br label %60

60:                                               ; preds = %67, %58
  %61 = phi i16 [ %59, %58 ], [ %68, %67 ]
  %62 = sext i16 %61 to i32
  %63 = sub i32 1, %62
  %64 = icmp ult i32 1, %62
  %65 = icmp ult i32 %63, 1
  %66 = or i1 %64, %65
  br i1 %66, label %139, label %67

67:                                               ; preds = %60
  %68 = add i16 %61, 1
  %69 = icmp slt i16 %68, 1
  br i1 %69, label %60, label %70

70:                                               ; preds = %67
  %71 = load i16, ptr %5, align 1
  br label %72

72:                                               ; preds = %79, %70
  %73 = phi i16 [ %71, %70 ], [ %80, %79 ]
  %74 = sext i16 %73 to i32
  %75 = sub i32 1, %74
  %76 = icmp ult i32 1, %74
  %77 = icmp ult i32 %75, 1
  %78 = or i1 %76, %77
  br i1 %78, label %139, label %79

79:                                               ; preds = %72
  %80 = add i16 %73, 1
  %81 = icmp slt i16 %80, 1
  br i1 %81, label %72, label %82

82:                                               ; preds = %79
  %83 = load i16, ptr %6, align 1
  br label %84

84:                                               ; preds = %91, %82
  %85 = phi i16 [ %83, %82 ], [ %92, %91 ]
  %86 = sext i16 %85 to i32
  %87 = sub i32 1, %86
  %88 = icmp ult i32 1, %86
  %89 = icmp ult i32 %87, 1
  %90 = or i1 %88, %89
  br i1 %90, label %139, label %91

91:                                               ; preds = %84
  %92 = add i16 %85, 1
  %93 = icmp slt i16 %92, 1
  br i1 %93, label %84, label %94

94:                                               ; preds = %91
  %95 = load i16, ptr %7, align 1
  br label %96

96:                                               ; preds = %103, %94
  %97 = phi i16 [ %95, %94 ], [ %104, %103 ]
  %98 = sext i16 %97 to i32
  %99 = sub i32 1, %98
  %100 = icmp ult i32 1, %98
  %101 = icmp ult i32 %99, 1
  %102 = or i1 %100, %101
  br i1 %102, label %139, label %103

103:                                              ; preds = %96
  %104 = add i16 %97, 1
  %105 = icmp slt i16 %104, 1
  br i1 %105, label %96, label %106

106:                                              ; preds = %103
  %107 = load i16, ptr %8, align 1
  br label %108

108:                                              ; preds = %115, %106
  %109 = phi i16 [ %107, %106 ], [ %116, %115 ]
  %110 = mul i16 %109, 4
  %111 = sub i16 24, %110
  %112 = icmp ult i16 24, %110
  %113 = icmp ult i16 %111, 4
  %114 = or i1 %112, %113
  br i1 %114, label %139, label %115

115:                                              ; preds = %108
  %116 = add i16 %109, 1
  %117 = icmp slt i16 %116, 6
  br i1 %117, label %108, label %118

118:                                              ; preds = %115
  %119 = load i16, ptr %9, align 1
  br label %120

120:                                              ; preds = %128, %118
  %121 = phi i16 [ %119, %118 ], [ %129, %128 ]
  %122 = sext i16 %121 to i32
  %123 = mul i32 %122, 2
  %124 = sub i32 4, %123
  %125 = icmp ult i32 4, %123
  %126 = icmp ult i32 %124, 2
  %127 = or i1 %125, %126
  br i1 %127, label %139, label %128

128:                                              ; preds = %120
  %129 = add i16 %121, 1
  %130 = icmp slt i16 %129, 2
  br i1 %130, label %120, label %131

131:                                              ; preds = %128
  %132 = load i16, ptr %10, align 1
  br label %133

133:                                              ; preds = %133, %131
  %134 = phi i16 [ %132, %131 ], [ %135, %133 ]
  %135 = add i16 %134, 1
  %136 = icmp slt i16 %135, 6
  br i1 %136, label %133, label %137

137:                                              ; preds = %133
  %138 = load i16, ptr %0, align 1
  ret i16 %138

139:                                              ; preds = %120, %108, %96, %84, %72, %60, %48, %35, %21, %13
  call void @llvm.trap() #2
  unreachable
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #1

attributes #0 = { nounwind }
attributes #1 = { cold noreturn nounwind }
attributes #2 = { noreturn nounwind }
