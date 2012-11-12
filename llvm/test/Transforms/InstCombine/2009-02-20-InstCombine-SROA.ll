; RUN: opt < %s -instcombine -scalarrepl -S | not grep " = alloca"
; rdar://6417724
; Instcombine shouldn't do anything to this function that prevents promoting the allocas inside it.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"

%"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >" = type { i32* }
%"struct.std::_Vector_base<int,std::allocator<int> >" = type { %"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl" }
%"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl" = type { i32*, i32*, i32* }
%"struct.std::bidirectional_iterator_tag" = type <{ i8 }>
%"struct.std::forward_iterator_tag" = type <{ i8 }>
%"struct.std::input_iterator_tag" = type <{ i8 }>
%"struct.std::random_access_iterator_tag" = type <{ i8 }>
%"struct.std::vector<int,std::allocator<int> >" = type { %"struct.std::_Vector_base<int,std::allocator<int> >" }

define i32* @_Z3fooRSt6vectorIiSaIiEE(%"struct.std::vector<int,std::allocator<int> >"* %X) {
entry:
  %0 = alloca %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"
  %__first_addr.i.i = alloca %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"
  %__last_addr.i.i = alloca %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"
  %unnamed_arg.i = alloca %"struct.std::bidirectional_iterator_tag", align 8
  %1 = alloca %"struct.std::bidirectional_iterator_tag"
  %__first_addr.i = alloca %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"
  %2 = alloca %"struct.std::bidirectional_iterator_tag"
  %3 = alloca %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"
  %4 = alloca i32
  %"alloca point" = bitcast i32 0 to i32
  store i32 42, i32* %4, align 4
  %5 = getelementptr %"struct.std::vector<int,std::allocator<int> >"* %X, i32 0, i32 0
  %6 = getelementptr %"struct.std::_Vector_base<int,std::allocator<int> >"* %5, i32 0, i32 0
  %7 = getelementptr %"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl"* %6, i32 0, i32 1
  %8 = load i32** %7, align 4
  %9 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %3, i32 0, i32 0
  store i32* %8, i32** %9, align 4
  %10 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %3, i32 0, i32 0
  %11 = load i32** %10, align 4
  %tmp2.i = ptrtoint i32* %11 to i32
  %tmp1.i = inttoptr i32 %tmp2.i to i32*
  %tmp3 = ptrtoint i32* %tmp1.i to i32
  %tmp2 = inttoptr i32 %tmp3 to i32*
  %12 = getelementptr %"struct.std::vector<int,std::allocator<int> >"* %X, i32 0, i32 0
  %13 = getelementptr %"struct.std::_Vector_base<int,std::allocator<int> >"* %12, i32 0, i32 0
  %14 = getelementptr %"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl"* %13, i32 0, i32 0
  %15 = load i32** %14, align 4
  %16 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %0, i32 0, i32 0
  store i32* %15, i32** %16, align 4
  %17 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %0, i32 0, i32 0
  %18 = load i32** %17, align 4
  %tmp2.i17 = ptrtoint i32* %18 to i32
  %tmp1.i18 = inttoptr i32 %tmp2.i17 to i32*
  %tmp8 = ptrtoint i32* %tmp1.i18 to i32
  %tmp6 = inttoptr i32 %tmp8 to i32*
  %19 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i, i32 0, i32 0
  store i32* %tmp6, i32** %19
  %20 = getelementptr %"struct.std::bidirectional_iterator_tag"* %1, i32 0, i32 0
  %21 = load i8* %20, align 1
  %22 = or i8 %21, 0
  %23 = or i8 %22, 0
  %24 = or i8 %23, 0
  %25 = getelementptr %"struct.std::bidirectional_iterator_tag"* %2, i32 0, i32 0
  store i8 0, i8* %25, align 1
  %elt.i = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i, i32 0, i32 0
  %val.i = load i32** %elt.i
  %tmp.i = bitcast %"struct.std::bidirectional_iterator_tag"* %unnamed_arg.i to i8*
  %tmp9.i = bitcast %"struct.std::bidirectional_iterator_tag"* %2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp.i, i8* %tmp9.i, i64 1, i32 1, i1 false)
  %26 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  store i32* %val.i, i32** %26
  %27 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__last_addr.i.i, i32 0, i32 0
  store i32* %tmp2, i32** %27
  %28 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__last_addr.i.i, i32 0, i32 0
  %29 = load i32** %28, align 4
  %30 = ptrtoint i32* %29 to i32
  %31 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %32 = load i32** %31, align 4
  %33 = ptrtoint i32* %32 to i32
  %34 = sub i32 %30, %33
  %35 = ashr i32 %34, 2
  %36 = ashr i32 %35, 2
  br label %bb12.i.i

bb.i.i:                                           ; preds = %bb12.i.i
  %37 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %38 = load i32** %37, align 4
  %39 = load i32* %38, align 4
  %40 = load i32* %4, align 4
  %41 = icmp eq i32 %39, %40
  %42 = zext i1 %41 to i8
  %toBool.i.i = icmp ne i8 %42, 0
  br i1 %toBool.i.i, label %bb1.i.i, label %bb2.i.i

bb1.i.i:                                          ; preds = %bb.i.i
  %43 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %44 = load i32** %43, align 4
  br label %_ZSt4findIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEEiET_S7_S7_RKT0_.exit

bb2.i.i:                                          ; preds = %bb.i.i
  %45 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %46 = load i32** %45, align 4
  %47 = getelementptr i32* %46, i64 1
  %48 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  store i32* %47, i32** %48, align 4
  %49 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %50 = load i32** %49, align 4
  %51 = load i32* %50, align 4
  %52 = load i32* %4, align 4
  %53 = icmp eq i32 %51, %52
  %54 = zext i1 %53 to i8
  %toBool3.i.i = icmp ne i8 %54, 0
  br i1 %toBool3.i.i, label %bb4.i.i, label %bb5.i.i

bb4.i.i:                                          ; preds = %bb2.i.i
  %55 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %56 = load i32** %55, align 4
  br label %_ZSt4findIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEEiET_S7_S7_RKT0_.exit

bb5.i.i:                                          ; preds = %bb2.i.i
  %57 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %58 = load i32** %57, align 4
  %59 = getelementptr i32* %58, i64 1
  %60 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  store i32* %59, i32** %60, align 4
  %61 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %62 = load i32** %61, align 4
  %63 = load i32* %62, align 4
  %64 = load i32* %4, align 4
  %65 = icmp eq i32 %63, %64
  %66 = zext i1 %65 to i8
  %toBool6.i.i = icmp ne i8 %66, 0
  br i1 %toBool6.i.i, label %bb7.i.i, label %bb8.i.i

bb7.i.i:                                          ; preds = %bb5.i.i
  %67 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %68 = load i32** %67, align 4
  br label %_ZSt4findIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEEiET_S7_S7_RKT0_.exit

bb8.i.i:                                          ; preds = %bb5.i.i
  %69 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %70 = load i32** %69, align 4
  %71 = getelementptr i32* %70, i64 1
  %72 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  store i32* %71, i32** %72, align 4
  %73 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %74 = load i32** %73, align 4
  %75 = load i32* %74, align 4
  %76 = load i32* %4, align 4
  %77 = icmp eq i32 %75, %76
  %78 = zext i1 %77 to i8
  %toBool9.i.i = icmp ne i8 %78, 0
  br i1 %toBool9.i.i, label %bb10.i.i, label %bb11.i.i

bb10.i.i:                                         ; preds = %bb8.i.i
  %79 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %80 = load i32** %79, align 4
  br label %_ZSt4findIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEEiET_S7_S7_RKT0_.exit

bb11.i.i:                                         ; preds = %bb8.i.i
  %81 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %82 = load i32** %81, align 4
  %83 = getelementptr i32* %82, i64 1
  %84 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  store i32* %83, i32** %84, align 4
  %85 = sub i32 %__trip_count.0.i.i, 1
  br label %bb12.i.i

bb12.i.i:                                         ; preds = %bb11.i.i, %entry
  %__trip_count.0.i.i = phi i32 [ %36, %entry ], [ %85, %bb11.i.i ]
  %86 = icmp sgt i32 %__trip_count.0.i.i, 0
  br i1 %86, label %bb.i.i, label %bb13.i.i

bb13.i.i:                                         ; preds = %bb12.i.i
  %87 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__last_addr.i.i, i32 0, i32 0
  %88 = load i32** %87, align 4
  %89 = ptrtoint i32* %88 to i32
  %90 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %91 = load i32** %90, align 4
  %92 = ptrtoint i32* %91 to i32
  %93 = sub i32 %89, %92
  %94 = ashr i32 %93, 2
  switch i32 %94, label %bb26.i.i [
    i32 1, label %bb22.i.i
    i32 2, label %bb18.i.i
    i32 3, label %bb14.i.i
  ]

bb14.i.i:                                         ; preds = %bb13.i.i
  %95 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %96 = load i32** %95, align 4
  %97 = load i32* %96, align 4
  %98 = load i32* %4, align 4
  %99 = icmp eq i32 %97, %98
  %100 = zext i1 %99 to i8
  %toBool15.i.i = icmp ne i8 %100, 0
  br i1 %toBool15.i.i, label %bb16.i.i, label %bb17.i.i

bb16.i.i:                                         ; preds = %bb14.i.i
  %101 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %102 = load i32** %101, align 4
  br label %_ZSt4findIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEEiET_S7_S7_RKT0_.exit

bb17.i.i:                                         ; preds = %bb14.i.i
  %103 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %104 = load i32** %103, align 4
  %105 = getelementptr i32* %104, i64 1
  %106 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  store i32* %105, i32** %106, align 4
  br label %bb18.i.i

bb18.i.i:                                         ; preds = %bb17.i.i, %bb13.i.i
  %107 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %108 = load i32** %107, align 4
  %109 = load i32* %108, align 4
  %110 = load i32* %4, align 4
  %111 = icmp eq i32 %109, %110
  %112 = zext i1 %111 to i8
  %toBool19.i.i = icmp ne i8 %112, 0
  br i1 %toBool19.i.i, label %bb20.i.i, label %bb21.i.i

bb20.i.i:                                         ; preds = %bb18.i.i
  %113 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %114 = load i32** %113, align 4
  br label %_ZSt4findIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEEiET_S7_S7_RKT0_.exit

bb21.i.i:                                         ; preds = %bb18.i.i
  %115 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %116 = load i32** %115, align 4
  %117 = getelementptr i32* %116, i64 1
  %118 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  store i32* %117, i32** %118, align 4
  br label %bb22.i.i

bb22.i.i:                                         ; preds = %bb21.i.i, %bb13.i.i
  %119 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %120 = load i32** %119, align 4
  %121 = load i32* %120, align 4
  %122 = load i32* %4, align 4
  %123 = icmp eq i32 %121, %122
  %124 = zext i1 %123 to i8
  %toBool23.i.i = icmp ne i8 %124, 0
  br i1 %toBool23.i.i, label %bb24.i.i, label %bb25.i.i

bb24.i.i:                                         ; preds = %bb22.i.i
  %125 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %126 = load i32** %125, align 4
  br label %_ZSt4findIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEEiET_S7_S7_RKT0_.exit

bb25.i.i:                                         ; preds = %bb22.i.i
  %127 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  %128 = load i32** %127, align 4
  %129 = getelementptr i32* %128, i64 1
  %130 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__first_addr.i.i, i32 0, i32 0
  store i32* %129, i32** %130, align 4
  br label %bb26.i.i

bb26.i.i:                                         ; preds = %bb25.i.i, %bb13.i.i
  %131 = getelementptr %"struct.__gnu_cxx::__normal_iterator<int*,std::vector<int, std::allocator<int> > >"* %__last_addr.i.i, i32 0, i32 0
  %132 = load i32** %131, align 4
  br label %_ZSt4findIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEEiET_S7_S7_RKT0_.exit

_ZSt4findIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEEiET_S7_S7_RKT0_.exit: ; preds = %bb26.i.i, %bb24.i.i, %bb20.i.i, %bb16.i.i, %bb10.i.i, %bb7.i.i, %bb4.i.i, %bb1.i.i
  %.0.0.i.i = phi i32* [ %132, %bb26.i.i ], [ %126, %bb24.i.i ], [ %114, %bb20.i.i ], [ %102, %bb16.i.i ], [ %80, %bb10.i.i ], [ %68, %bb7.i.i ], [ %56, %bb4.i.i ], [ %44, %bb1.i.i ]
  %tmp2.i.i = ptrtoint i32* %.0.0.i.i to i32
  %tmp1.i.i = inttoptr i32 %tmp2.i.i to i32*
  %tmp4.i = ptrtoint i32* %tmp1.i.i to i32
  %tmp3.i = inttoptr i32 %tmp4.i to i32*
  %tmp8.i = ptrtoint i32* %tmp3.i to i32
  %tmp6.i = inttoptr i32 %tmp8.i to i32*
  %tmp12 = ptrtoint i32* %tmp6.i to i32
  %tmp10 = inttoptr i32 %tmp12 to i32*
  %tmp16 = ptrtoint i32* %tmp10 to i32
  br label %return

return:                                           ; preds = %_ZSt4findIN9__gnu_cxx17__normal_iteratorIPiSt6vectorIiSaIiEEEEiET_S7_S7_RKT0_.exit
  %tmp14 = inttoptr i32 %tmp16 to i32*
  ret i32* %tmp14
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
