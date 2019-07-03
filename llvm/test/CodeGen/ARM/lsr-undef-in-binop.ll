; REQUIRES: arm-registered-target
; RUN: opt -S -loop-reduce %s -o - | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8-unknown-hurd-eabihf"

%"class.std::__1::vector.182" = type { %"class.std::__1::__vector_base.183" }
%"class.std::__1::__vector_base.183" = type { i8*, i8*, %"class.std::__1::__compressed_pair.184" }
%"class.std::__1::__compressed_pair.184" = type { %"struct.std::__1::__compressed_pair_elem.185" }
%"struct.std::__1::__compressed_pair_elem.185" = type { i8* }
%"class.std::__1::__vector_base_common" = type { i8 }

$vector_insert = comdat any

declare i8* @Allocate(i32) local_unnamed_addr
declare void @Free(i8*) local_unnamed_addr
declare void @_ZNKSt3__120__vector_base_commonILb1EE20__throw_length_errorEv(%"class.std::__1::__vector_base_common"*) local_unnamed_addr
declare i8* @memmove(i8*, i8*, i32) local_unnamed_addr

; Function Attrs: noimplicitfloat nounwind uwtable
define linkonce_odr i32 @vector_insert(%"class.std::__1::vector.182"*, [1 x i32], i8*, i8*) local_unnamed_addr #1 comdat align 2 {
; CHECK-LABEL: vector_insert
  %5 = extractvalue [1 x i32] %1, 0
  %6 = getelementptr inbounds %"class.std::__1::vector.182", %"class.std::__1::vector.182"* %0, i32 0, i32 0, i32 0
  %7 = load i8*, i8** %6, align 4
; CHECK: [[LOAD:%[0-9]+]] = load i8*, i8**
  %8 = bitcast %"class.std::__1::vector.182"* %0 to i32*
  %9 = ptrtoint i8* %7 to i32
; CHECK: [[NEW_CAST:%[0-9]+]] = ptrtoint i8* [[LOAD]] to i32
; CHECK: [[OLD_CAST:%[0-9]+]] = ptrtoint i8* [[LOAD]] to i32
  %10 = sub i32 %5, %9
  %11 = getelementptr inbounds i8, i8* %7, i32 %10
  %12 = ptrtoint i8* %3 to i32
  %13 = ptrtoint i8* %2 to i32
  %14 = sub i32 %12, %13
  %15 = icmp sgt i32 %14, 0
  br i1 %15, label %18, label %16

; <label>:16:                                     ; preds = %4
  %17 = ptrtoint i8* %11 to i32
  br label %148

; <label>:18:                                     ; preds = %4
  %19 = getelementptr inbounds %"class.std::__1::vector.182", %"class.std::__1::vector.182"* %0, i32 0, i32 0, i32 2, i32 0, i32 0
  %20 = bitcast i8** %19 to i32*
  %21 = load i32, i32* %20, align 4
  %22 = getelementptr inbounds %"class.std::__1::vector.182", %"class.std::__1::vector.182"* %0, i32 0, i32 0, i32 1
  %23 = load i8*, i8** %22, align 4
  %24 = ptrtoint i8* %23 to i32
  %25 = sub i32 %21, %24
  %26 = icmp sgt i32 %14, %25
  %27 = bitcast i8** %22 to i32*
  br i1 %26, label %77, label %28

; <label>:28:                                     ; preds = %18
  %29 = ptrtoint i8* %11 to i32
  %30 = sub i32 %24, %29
  %31 = icmp sgt i32 %14, %30
  br i1 %31, label %32, label %48

; <label>:32:                                     ; preds = %28
  %33 = getelementptr inbounds i8, i8* %2, i32 %30
  %34 = icmp eq i8* %33, %3
  br i1 %34, label %43, label %35

; <label>:35:                                     ; preds = %32, %35
  %36 = phi i8* [ %41, %35 ], [ %23, %32 ]
  %37 = phi i8* [ %39, %35 ], [ %33, %32 ]
  %38 = load i8, i8* %37, align 1
  store i8 %38, i8* %36, align 1
  %39 = getelementptr inbounds i8, i8* %37, i32 1
  %40 = load i8*, i8** %22, align 4
  %41 = getelementptr inbounds i8, i8* %40, i32 1
  store i8* %41, i8** %22, align 4
  %42 = icmp eq i8* %39, %3
  br i1 %42, label %43, label %35

; <label>:43:                                     ; preds = %35, %32
  %44 = phi i8* [ %23, %32 ], [ %41, %35 ]
  %45 = icmp sgt i32 %30, 0
  br i1 %45, label %46, label %148

; <label>:46:                                     ; preds = %43
  %47 = ptrtoint i8* %44 to i32
  br label %48

; <label>:48:                                     ; preds = %46, %28
  %49 = phi i32 [ %47, %46 ], [ %24, %28 ]
  %50 = phi i8* [ %44, %46 ], [ %23, %28 ]
  %51 = phi i8* [ %33, %46 ], [ %3, %28 ]
  %52 = getelementptr inbounds i8, i8* %11, i32 %14
  %53 = ptrtoint i8* %52 to i32
  %54 = sub i32 %49, %53
  %55 = getelementptr inbounds i8, i8* %11, i32 %54
  %56 = icmp ult i8* %55, %23
  br i1 %56, label %63, label %57

; <label>:57:                                     ; preds = %63, %48
  %58 = icmp eq i32 %54, 0
  br i1 %58, label %71, label %59

; <label>:59:                                     ; preds = %57
  %60 = sub i32 0, %54
  %61 = getelementptr inbounds i8, i8* %50, i32 %60
  %62 = tail call i8* @memmove(i8* %61, i8* %11, i32 %54) #13
  br label %71

; <label>:63:                                     ; preds = %48, %63
  %64 = phi i8* [ %69, %63 ], [ %50, %48 ]
  %65 = phi i8* [ %67, %63 ], [ %55, %48 ]
  %66 = load i8, i8* %65, align 1
  store i8 %66, i8* %64, align 1
  %67 = getelementptr inbounds i8, i8* %65, i32 1
  %68 = load i8*, i8** %22, align 4
  %69 = getelementptr inbounds i8, i8* %68, i32 1
  store i8* %69, i8** %22, align 4
  %70 = icmp eq i8* %67, %23
  br i1 %70, label %57, label %63

; <label>:71:                                     ; preds = %57, %59
  %72 = ptrtoint i8* %51 to i32
  %73 = sub i32 %72, %13
  %74 = icmp eq i32 %73, 0
  br i1 %74, label %148, label %75

; <label>:75:                                     ; preds = %71
  %76 = tail call i8* @memmove(i8* %11, i8* %2, i32 %73) #13
  br label %148

; <label>:77:                                     ; preds = %18
  %78 = sub i32 %24, %9
  %79 = add i32 %78, %14
  %80 = icmp slt i32 %79, 0
  br i1 %80, label %81, label %83

; <label>:81:                                     ; preds = %77
  %82 = bitcast %"class.std::__1::vector.182"* %0 to %"class.std::__1::__vector_base_common"*
  tail call void @_ZNKSt3__120__vector_base_commonILb1EE20__throw_length_errorEv(%"class.std::__1::__vector_base_common"* %82) #15
  unreachable

; <label>:83:                                     ; preds = %77
  %84 = sub i32 %21, %9
  %85 = icmp ult i32 %84, 1073741823
  br i1 %85, label %86, label %91

; <label>:86:                                     ; preds = %83
  %87 = shl i32 %84, 1
  %88 = icmp ult i32 %87, %79
  %89 = select i1 %88, i32 %79, i32 %87
  %90 = icmp eq i32 %89, 0
  br i1 %90, label %94, label %91

; <label>:91:                                     ; preds = %83, %86
  %92 = phi i32 [ %89, %86 ], [ 2147483647, %83 ]
  %93 = tail call i8* @Allocate(i32 %92) #13
  br label %94

; <label>:94:                                     ; preds = %86, %91
  %95 = phi i32 [ %92, %91 ], [ 0, %86 ]
  %96 = phi i8* [ %93, %91 ], [ null, %86 ]
  %97 = getelementptr inbounds i8, i8* %96, i32 %10
  %98 = ptrtoint i8* %97 to i32
  %99 = getelementptr inbounds i8, i8* %96, i32 %95
  %100 = ptrtoint i8* %99 to i32
  %101 = icmp eq i8* %2, %3
  br i1 %101, label %111, label %102

; <label>:102:                                    ; preds = %94, %102
  %103 = phi i8* [ %106, %102 ], [ %97, %94 ]
  %104 = phi i8* [ %107, %102 ], [ %2, %94 ]
  %105 = load i8, i8* %104, align 1
  store i8 %105, i8* %103, align 1
  %106 = getelementptr inbounds i8, i8* %103, i32 1
  %107 = getelementptr inbounds i8, i8* %104, i32 1
  %108 = icmp eq i8* %107, %3
  br i1 %108, label %109, label %102

; <label>:109:                                    ; preds = %102
  %110 = ptrtoint i8* %106 to i32
  br label %111

; <label>:111:                                    ; preds = %109, %94
  %112 = phi i32 [ %98, %94 ], [ %110, %109 ]
  %113 = load i8*, i8** %6, align 4
  %114 = icmp eq i8* %113, %11
  br i1 %114, label %124, label %115

; CHECK-LABEL: .preheader:
; CHECK-NEXT: sub i32 [[OLD_CAST]], [[NEW_CAST]]
; <label>:115:                                    ; preds = %111, %115
  %116 = phi i8* [ %118, %115 ], [ %97, %111 ]
  %117 = phi i8* [ %119, %115 ], [ %11, %111 ]
  %118 = getelementptr inbounds i8, i8* %116, i32 -1
  %119 = getelementptr inbounds i8, i8* %117, i32 -1
  %120 = load i8, i8* %119, align 1
  store i8 %120, i8* %118, align 1
  %121 = icmp eq i8* %119, %113
  br i1 %121, label %122, label %115

; <label>:122:                                    ; preds = %115
  %123 = ptrtoint i8* %118 to i32
  br label %124

; <label>:124:                                    ; preds = %122, %111
  %125 = phi i32 [ %98, %111 ], [ %123, %122 ]
  %126 = phi i8* [ %97, %111 ], [ %118, %122 ]
  %127 = load i8*, i8** %22, align 4
  %128 = icmp eq i8* %127, %11
  br i1 %128, label %129, label %131

; <label>:129:                                    ; preds = %124
  %130 = ptrtoint i8* %126 to i32
  br label %142

; <label>:131:                                    ; preds = %124
  %132 = inttoptr i32 %112 to i8*
  br label %133

; <label>:133:                                    ; preds = %133, %131
  %134 = phi i8* [ %138, %133 ], [ %132, %131 ]
  %135 = phi i8* [ %137, %133 ], [ %11, %131 ]
  %136 = load i8, i8* %135, align 1
  store i8 %136, i8* %134, align 1
  %137 = getelementptr inbounds i8, i8* %135, i32 1
  %138 = getelementptr inbounds i8, i8* %134, i32 1
  %139 = icmp eq i8* %137, %127
  br i1 %139, label %140, label %133

; <label>:140:                                    ; preds = %133
  %141 = ptrtoint i8* %138 to i32
  br label %142

; <label>:142:                                    ; preds = %140, %129
  %143 = phi i32 [ %112, %129 ], [ %141, %140 ]
  %144 = phi i32 [ %130, %129 ], [ %125, %140 ]
  %145 = load i8*, i8** %6, align 4
  store i32 %144, i32* %8, align 4
  store i32 %143, i32* %27, align 4
  store i32 %100, i32* %20, align 4
  %146 = icmp eq i8* %145, null
  br i1 %146, label %148, label %147

; <label>:147:                                    ; preds = %142
  tail call void @Free(i8* nonnull %145) #13
  br label %148

; <label>:148:                                    ; preds = %16, %147, %142, %43, %71, %75
  %149 = phi i32 [ %17, %16 ], [ %98, %147 ], [ %98, %142 ], [ %29, %43 ], [ %29, %71 ], [ %29, %75 ]
  ret i32 %149
}

