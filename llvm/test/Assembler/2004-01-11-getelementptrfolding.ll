; RUN: llvm-as < %s | llvm-dis | \
; RUN:   not grep {getelementptr.*getelementptr}

%struct.TTriangleItem = type { i8*, i8*, [3 x %struct.TUVVertex] }
%struct.TUVVertex = type { i16, i16, i16, i16 }
@data_triangleItems = internal constant [2908 x %struct.TTriangleItem] zeroinitializer; <[2908 x %struct.TTriangleItem]*> [#uses=2]

define void @foo() {
        store i16 0, i16* getelementptr ([2908 x %struct.TTriangleItem]* @data_triangleItems, i64 0, i64 0, i32 2, i64 0, i32 0)
        ret void
}

