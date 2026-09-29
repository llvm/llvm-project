; RUN: llc < %s

define i8 @test1(double %X) {
	%tmp.1 = fptosi double %X to i8		; <i8> [#uses=1]
	ret i8 %tmp.1
}

define i16 @test2(double %X) {
	%tmp.1 = fptosi double %X to i16		; <i16> [#uses=1]
	ret i16 %tmp.1
}

define i32 @test3(double %X) {
	%tmp.1 = fptosi double %X to i32		; <i32> [#uses=1]
	ret i32 %tmp.1
}

define i64 @test4(double %X) {
	%tmp.1 = fptosi double %X to i64		; <i64> [#uses=1]
	ret i64 %tmp.1
}

define i8 @test1u(double %X) {
	%tmp.1 = fptoui double %X to i8		; <i8> [#uses=1]
	ret i8 %tmp.1
}

define i16 @test2u(double %X) {
	%tmp.1 = fptoui double %X to i16		; <i16> [#uses=1]
	ret i16 %tmp.1
}

define i32 @test3u(double %X) {
	%tmp.1 = fptoui double %X to i32		; <i32> [#uses=1]
	ret i32 %tmp.1
}

define i64 @test4u(double %X) {
	%tmp.1 = fptoui double %X to i64		; <i64> [#uses=1]
	ret i64 %tmp.1
}

define i8 @test1f(float %X) {
	%tmp.1 = fptosi float %X to i8		; <i8> [#uses=1]
	ret i8 %tmp.1
}

define i16 @test2f(float %X) {
	%tmp.1 = fptosi float %X to i16		; <i16> [#uses=1]
	ret i16 %tmp.1
}

define i32 @test3f(float %X) {
	%tmp.1 = fptosi float %X to i32		; <i32> [#uses=1]
	ret i32 %tmp.1
}

define i64 @test4f(float %X) {
	%tmp.1 = fptosi float %X to i64		; <i64> [#uses=1]
	ret i64 %tmp.1
}

define i8 @test1uf(float %X) {
	%tmp.1 = fptoui float %X to i8		; <i8> [#uses=1]
	ret i8 %tmp.1
}

define i16 @test2uf(float %X) {
	%tmp.1 = fptoui float %X to i16		; <i16> [#uses=1]
	ret i16 %tmp.1
}

define i32 @test3uf(float %X) {
	%tmp.1 = fptoui float %X to i32		; <i32> [#uses=1]
	ret i32 %tmp.1
}

define i64 @test4uf(float %X) {
	%tmp.1 = fptoui float %X to i64		; <i64> [#uses=1]
	ret i64 %tmp.1
}
