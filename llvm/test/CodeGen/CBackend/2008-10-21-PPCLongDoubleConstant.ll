; RUN: llvm-as < %s | llc -march=c
; PR2907
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-apple-darwin9.5"
	%"struct.Point<0>" = type { %"struct.Tensor<1,0>" }
	%"struct.QGauss2<1>" = type { %"struct.Quadrature<0>" }
	%"struct.Quadrature<0>" = type { %struct.Subscriptor, i32, %"struct.std::vector<Point<0>,std::allocator<Point<0> > >", %"struct.std::vector<double,std::allocator<double> >" }
	%struct.Subscriptor = type { i32 (...)**, i32, %"struct.std::type_info"* }
	%"struct.Tensor<1,0>" = type { [1 x double] }
	%"struct.std::_Vector_base<Point<0>,std::allocator<Point<0> > >" = type { %"struct.std::_Vector_base<Point<0>,std::allocator<Point<0> > >::_Vector_impl" }
	%"struct.std::_Vector_base<Point<0>,std::allocator<Point<0> > >::_Vector_impl" = type { %"struct.Point<0>"*, %"struct.Point<0>"*, %"struct.Point<0>"* }
	%"struct.std::_Vector_base<double,std::allocator<double> >" = type { %"struct.std::_Vector_base<double,std::allocator<double> >::_Vector_impl" }
	%"struct.std::_Vector_base<double,std::allocator<double> >::_Vector_impl" = type { double*, double*, double* }
	%"struct.std::type_info" = type { i32 (...)**, i8* }
	%"struct.std::vector<Point<0>,std::allocator<Point<0> > >" = type { %"struct.std::_Vector_base<Point<0>,std::allocator<Point<0> > >" }
	%"struct.std::vector<double,std::allocator<double> >" = type { %"struct.std::_Vector_base<double,std::allocator<double> >" }

define fastcc void @_ZN6QGaussILi1EEC1Ej(%"struct.QGauss2<1>"* %this, i32 %n) {
entry:
	br label %bb4

bb4:		; preds = %bb5.split, %bb4, %entry
	%0 = fcmp ogt ppc_fp128 0xM00000000000000000000000000000000, select (i1 fcmp olt (ppc_fp128 fpext (double 0x3C447AE147AE147B to ppc_fp128), ppc_fp128 mul (ppc_fp128 0xM00000000000000010000000000000000, ppc_fp128 0xM40140000000000000000000000000000)), ppc_fp128 mul (ppc_fp128 0xM00000000000000010000000000000000, ppc_fp128 0xM40140000000000000000000000000000), ppc_fp128 fpext (double 0x3C447AE147AE147B to ppc_fp128))		; <i1> [#uses=1]
	br i1 %0, label %bb4, label %bb5.split

bb5.split:		; preds = %bb4
	%1 = getelementptr double* null, i32 0		; <double*> [#uses=0]
	br label %bb4
}
