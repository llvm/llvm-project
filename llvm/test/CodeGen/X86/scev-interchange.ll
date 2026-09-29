; RUN: llc < %s -mtriple=x86_64-linux

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
	%"struct.DataOutBase::GmvFlags" = type { i32 }
	%"struct.FE_DGPNonparametric<3>" = type { [1156 x i8], i32, %"struct.PolynomialSpace<1>" }
	%"struct.FiniteElementData<1>" = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
	%struct.Line = type { [2 x i32] }
	%"struct.PolynomialSpace<1>" = type { %"struct.std::vector<Polynomials::Polynomial<double>,std::allocator<Polynomials::Polynomial<double> > >", i32, %"struct.std::vector<int,std::allocator<int> >", %"struct.std::vector<int,std::allocator<int> >" }
	%"struct.Polynomials::Polynomial<double>" = type { %struct.Subscriptor, %"struct.std::vector<double,std::allocator<double> >" }
	%struct.Subscriptor = type { ptr, i32, ptr }
	%"struct.TableBase<2,double>" = type { %struct.Subscriptor, ptr, i32, %"struct.TableIndices<2>" }
	%"struct.TableIndices<2>" = type { %struct.Line }
	%"struct.std::_Bit_const_iterator" = type { %"struct.std::_Bit_iterator_base" }
	%"struct.std::_Bit_iterator_base" = type { ptr, i32 }
	%"struct.std::_Bvector_base<std::allocator<bool> >" = type { %"struct.std::_Bvector_base<std::allocator<bool> >::_Bvector_impl" }
	%"struct.std::_Bvector_base<std::allocator<bool> >::_Bvector_impl" = type { %"struct.std::_Bit_const_iterator", %"struct.std::_Bit_const_iterator", ptr }
	%"struct.std::_Vector_base<Polynomials::Polynomial<double>,std::allocator<Polynomials::Polynomial<double> > >" = type { %"struct.std::_Vector_base<Polynomials::Polynomial<double>,std::allocator<Polynomials::Polynomial<double> > >::_Vector_impl" }
	%"struct.std::_Vector_base<Polynomials::Polynomial<double>,std::allocator<Polynomials::Polynomial<double> > >::_Vector_impl" = type { ptr, ptr, ptr }
	%"struct.std::_Vector_base<double,std::allocator<double> >" = type { %"struct.std::_Vector_base<double,std::allocator<double> >::_Vector_impl" }
	%"struct.std::_Vector_base<double,std::allocator<double> >::_Vector_impl" = type { ptr, ptr, ptr }
	%"struct.std::_Vector_base<int,std::allocator<int> >" = type { %"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl" }
	%"struct.std::_Vector_base<int,std::allocator<int> >::_Vector_impl" = type { ptr, ptr, ptr }
	%"struct.std::_Vector_base<std::vector<bool, std::allocator<bool> >,std::allocator<std::vector<bool, std::allocator<bool> > > >" = type { %"struct.std::_Vector_base<std::vector<bool, std::allocator<bool> >,std::allocator<std::vector<bool, std::allocator<bool> > > >::_Vector_impl" }
	%"struct.std::_Vector_base<std::vector<bool, std::allocator<bool> >,std::allocator<std::vector<bool, std::allocator<bool> > > >::_Vector_impl" = type { ptr, ptr, ptr }
	%"struct.std::type_info" = type { ptr, ptr }
	%"struct.std::vector<Polynomials::Polynomial<double>,std::allocator<Polynomials::Polynomial<double> > >" = type { %"struct.std::_Vector_base<Polynomials::Polynomial<double>,std::allocator<Polynomials::Polynomial<double> > >" }
	%"struct.std::vector<bool,std::allocator<bool> >" = type { %"struct.std::_Bvector_base<std::allocator<bool> >" }
	%"struct.std::vector<double,std::allocator<double> >" = type { %"struct.std::_Vector_base<double,std::allocator<double> >" }
	%"struct.std::vector<int,std::allocator<int> >" = type { %"struct.std::_Vector_base<int,std::allocator<int> >" }
	%"struct.std::vector<std::vector<bool, std::allocator<bool> >,std::allocator<std::vector<bool, std::allocator<bool> > > >" = type { %"struct.std::_Vector_base<std::vector<bool, std::allocator<bool> >,std::allocator<std::vector<bool, std::allocator<bool> > > >" }

declare void @_Unwind_Resume(ptr)

declare ptr @_Znwm(i64)

declare fastcc void @_ZNSt6vectorIjSaIjEEaSERKS1_(ptr, ptr)

declare fastcc void @_ZN9TableBaseILi2EdE6reinitERK12TableIndicesILi2EE(ptr nocapture, i32, i32)

declare fastcc void @_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_(ptr nocapture, i64, ptr nocapture)

declare fastcc void @_ZNSt6vectorIS_IbSaIbEESaIS1_EEC2EmRKS1_RKS2_(ptr nocapture, i64, ptr nocapture)

declare fastcc void @_ZNSt6vectorIN11Polynomials10PolynomialIdEESaIS2_EED1Ev(ptr nocapture)

declare fastcc void @_ZN24TensorProductPolynomialsILi3EEC2IN11Polynomials10PolynomialIdEEEERKSt6vectorIT_SaIS6_EE(ptr nocapture, ptr nocapture)

declare fastcc void @_ZN7FE_PolyI24TensorProductPolynomialsILi3EELi3EEC2EjRKS1_RK17FiniteElementDataILi3EERKSt6vectorIbSaIbEERKS9_ISB_SaISB_EE(ptr, i32, ptr nocapture, ptr nocapture, ptr nocapture, ptr nocapture)

declare fastcc void @_ZN11FE_Q_Helper12_GLOBAL__N_116invert_numberingERKSt6vectorIjSaIjEE(ptr noalias nocapture sret(%"struct.std::vector<int,std::allocator<int> >"), ptr nocapture)

declare fastcc void @_ZN4FE_QILi3EE14get_dpo_vectorEj(ptr noalias nocapture sret(%"struct.std::vector<int,std::allocator<int> >"), i32)

define fastcc void @_ZN4FE_QILi3EEC1Ej(i32 %degree) personality ptr @__gxx_personality_v0 {
entry:
	invoke fastcc void @_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_(ptr undef, i64 1, ptr undef)
			to label %invcont.i unwind label %lpad.i

invcont.i:		; preds = %entry
	invoke fastcc void @_ZN4FE_QILi3EE14get_dpo_vectorEj(ptr noalias sret(%"struct.std::vector<int,std::allocator<int> >") undef, i32 %degree)
			to label %invcont1.i unwind label %lpad120.i

invcont1.i:		; preds = %invcont.i
	invoke fastcc void @_ZNSt6vectorIS_IbSaIbEESaIS1_EEC2EmRKS1_RKS2_(ptr undef, i64 undef, ptr undef)
			to label %invcont3.i unwind label %lpad124.i

invcont3.i:		; preds = %invcont1.i
	invoke fastcc void @_ZN4FE_QILi3EE14get_dpo_vectorEj(ptr noalias sret(%"struct.std::vector<int,std::allocator<int> >") undef, i32 %degree)
			to label %invcont4.i unwind label %lpad128.i

invcont4.i:		; preds = %invcont3.i
	invoke fastcc void @_ZNSt6vectorIbSaIbEEC1EmRKbRKS0_(ptr undef, i64 undef, ptr undef)
			to label %invcont6.i unwind label %lpad132.i

invcont6.i:		; preds = %invcont4.i
	invoke fastcc void @_ZN4FE_QILi3EE14get_dpo_vectorEj(ptr noalias sret(%"struct.std::vector<int,std::allocator<int> >") undef, i32 %degree)
			to label %invcont7.i unwind label %lpad136.i

invcont7.i:		; preds = %invcont6.i
	invoke fastcc void @_ZN11Polynomials19LagrangeEquidistant23generate_complete_basisEj(ptr noalias sret(%"struct.std::vector<Polynomials::Polynomial<double>,std::allocator<Polynomials::Polynomial<double> > >") undef, i32 %degree)
			to label %invcont9.i unwind label %lpad140.i

invcont9.i:		; preds = %invcont7.i
	invoke fastcc void @_ZN24TensorProductPolynomialsILi3EEC2IN11Polynomials10PolynomialIdEEEERKSt6vectorIT_SaIS6_EE(ptr undef, ptr undef)
			to label %invcont10.i unwind label %lpad144.i

invcont10.i:		; preds = %invcont9.i
	invoke fastcc void @_ZN7FE_PolyI24TensorProductPolynomialsILi3EELi3EEC2EjRKS1_RK17FiniteElementDataILi3EERKSt6vectorIbSaIbEERKS9_ISB_SaISB_EE(ptr undef, i32 %degree, ptr undef, ptr undef, ptr undef, ptr undef)
			to label %bb14.i unwind label %lpad148.i

bb14.i:		; preds = %invcont10.i
	br i1 false, label %bb3.i164.i, label %bb.i.i.i.i160.i

bb.i.i.i.i160.i:		; preds = %bb14.i
	unreachable

bb3.i164.i:		; preds = %bb14.i
	br i1 undef, label %bb10.i168.i, label %bb.i.i.i20.i166.i

bb.i.i.i20.i166.i:		; preds = %bb3.i164.i
	unreachable

bb10.i168.i:		; preds = %bb3.i164.i
	invoke fastcc void @_ZNSt6vectorIN11Polynomials10PolynomialIdEESaIS2_EED1Ev(ptr undef)
			to label %bb21.i unwind label %lpad144.i

bb21.i:		; preds = %bb10.i168.i
	invoke fastcc void @_ZNSt6vectorIN11Polynomials10PolynomialIdEESaIS2_EED1Ev(ptr undef)
			to label %bb28.i unwind label %lpad140.i

bb28.i:		; preds = %bb21.i
	br i1 undef, label %bb35.i, label %bb.i.i.i175.i

bb.i.i.i175.i:		; preds = %bb28.i
	br label %bb35.i

bb35.i:		; preds = %bb.i.i.i175.i, %bb28.i
	br i1 undef, label %bb42.i, label %bb.i.i.i205.i

bb.i.i.i205.i:		; preds = %bb35.i
	unreachable

bb42.i:		; preds = %bb35.i
	br i1 undef, label %bb47.i, label %bb.i.i.i213.i

bb.i.i.i213.i:		; preds = %bb42.i
	unreachable

bb47.i:		; preds = %bb42.i
	br i1 undef, label %bb59.i, label %bb.i.i.i247.i

bb.i.i.i247.i:		; preds = %bb47.i
	unreachable

bb59.i:		; preds = %bb47.i
	br i1 undef, label %bb66.i, label %bb.i.i.i255.i

bb.i.i.i255.i:		; preds = %bb59.i
	unreachable

bb66.i:		; preds = %bb59.i
	br i1 undef, label %bb71.i, label %bb.i.i.i262.i

bb.i.i.i262.i:		; preds = %bb66.i
	br label %bb71.i

bb71.i:		; preds = %bb.i.i.i262.i, %bb66.i
	%tmp11.i.i29.i.i.i.i.i.i = invoke ptr @_Znwm(i64 12)
			to label %_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i.i.i unwind label %lpad.i.i.i.i.i.i		; <ptr> [#uses=0]

lpad.i.i.i.i.i.i:		; preds = %bb71.i
        %exn.i.i.i.i.i.i = landingpad {ptr, i32}
                 cleanup
	unreachable

_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i.i.i:		; preds = %bb71.i
	br i1 undef, label %_ZNSt6vectorIjSaIjEED1Ev.exit.i.i, label %bb.i.i.i.i94.i

bb.i.i.i.i94.i:		; preds = %_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i.i.i
	unreachable

_ZNSt6vectorIjSaIjEED1Ev.exit.i.i:		; preds = %_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i.i.i
	%tmp11.i.i29.i.i.i.i5.i.i = invoke ptr @_Znwm(i64 undef)
			to label %_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i12.i.i unwind label %lpad.i.i.i.i8.i.i		; <ptr> [#uses=0]

lpad.i.i.i.i8.i.i:		; preds = %_ZNSt6vectorIjSaIjEED1Ev.exit.i.i
        %exn.i.i.i.i8.i.i = landingpad {ptr, i32}
                 cleanup
	invoke void @_Unwind_Resume(ptr undef)
			to label %.noexc.i9.i.i unwind label %lpad.i19.i.i

.noexc.i9.i.i:		; preds = %lpad.i.i.i.i8.i.i
	unreachable

_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i12.i.i:		; preds = %_ZNSt6vectorIjSaIjEED1Ev.exit.i.i
	br i1 undef, label %bb50.i.i.i, label %bb.i.i.i.i.i.i.i.i.i.i

bb.i.i.i.i.i.i.i.i.i.i:		; preds = %bb.i.i.i.i.i.i.i.i.i.i, %_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i12.i.i
	br i1 undef, label %bb50.i.i.i, label %bb.i.i.i.i.i.i.i.i.i.i

bb50.i.i.i:		; preds = %bb.i.i.i.i.i.i.i.i.i.i, %_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i12.i.i
	invoke fastcc void @_ZN11FE_Q_Helper12_GLOBAL__N_116invert_numberingERKSt6vectorIjSaIjEE(ptr noalias sret(%"struct.std::vector<int,std::allocator<int> >") undef, ptr undef)
			to label %bb83.i unwind label %lpad188.i

lpad.i19.i.i:		; preds = %lpad.i.i.i.i8.i.i
        %exn.i19.i.i = landingpad {ptr, i32}
                 cleanup
	unreachable

bb83.i:		; preds = %bb50.i.i.i
	br i1 undef, label %invcont84.i, label %bb.i.i.i221.i

bb.i.i.i221.i:		; preds = %bb83.i
	unreachable

invcont84.i:		; preds = %bb83.i
	%tmp11.i.i29.i.i.i.i.i = invoke ptr @_Znwm(i64 undef)
			to label %_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i.i unwind label %lpad.i.i.i.i315.i		; <ptr> [#uses=0]

lpad.i.i.i.i315.i:		; preds = %invcont84.i
        %exn.i.i.i.i315.i = landingpad {ptr, i32}
                 cleanup
	invoke void @_Unwind_Resume(ptr undef)
			to label %.noexc.i316.i unwind label %lpad.i352.i

.noexc.i316.i:		; preds = %lpad.i.i.i.i315.i
	unreachable

_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i.i:		; preds = %invcont84.i
	br i1 undef, label %bb50.i.i, label %bb.i.i.i.i.i.i.i.i320.i

bb.i.i.i.i.i.i.i.i320.i:		; preds = %bb.i.i.i.i.i.i.i.i320.i, %_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i.i
	br i1 undef, label %bb50.i.i, label %bb.i.i.i.i.i.i.i.i320.i

bb50.i.i:		; preds = %bb.i.i.i.i.i.i.i.i320.i, %_ZNSt12_Vector_baseIjSaIjEEC2EmRKS0_.exit.i.i.i.i
	invoke fastcc void @_ZN11FE_Q_Helper12_GLOBAL__N_116invert_numberingERKSt6vectorIjSaIjEE(ptr noalias sret(%"struct.std::vector<int,std::allocator<int> >") undef, ptr undef)
			to label %invcont86.i unwind label %lpad200.i

lpad.i352.i:		; preds = %lpad.i.i.i.i315.i
        %exn.i352.i = landingpad {ptr, i32}
                 cleanup
	unreachable

invcont86.i:		; preds = %bb50.i.i
	invoke fastcc void @_ZNSt6vectorIjSaIjEEaSERKS1_(ptr undef, ptr undef)
			to label %.noexc380.i unwind label %lpad204.i

.noexc380.i:		; preds = %invcont86.i
	br i1 undef, label %bb100.i, label %bb.i.i.i198.i

bb.i.i.i198.i:		; preds = %.noexc380.i
	unreachable

bb100.i:		; preds = %.noexc380.i
	br i1 undef, label %invcont101.i, label %bb.i.i.i190.i

bb.i.i.i190.i:		; preds = %bb100.i
	unreachable

invcont101.i:		; preds = %bb100.i
	invoke fastcc void @_ZN9TableBaseILi2EdE6reinitERK12TableIndicesILi2EE(ptr undef, i32 undef, i32 undef)
			to label %_ZN10FullMatrixIdEC1Ejj.exit.i.i unwind label %lpad.i.i.i.i.i

lpad.i.i.i.i.i:		; preds = %invcont101.i
        %exn.i.i.i.i.i = landingpad {ptr, i32}
                 cleanup
	unreachable

_ZN10FullMatrixIdEC1Ejj.exit.i.i:		; preds = %invcont101.i
	invoke fastcc void @_ZN9TableBaseILi2EdE6reinitERK12TableIndicesILi2EE(ptr undef, i32 undef, i32 undef)
			to label %_ZN10FullMatrixIdEC1Ejj.exit28.i.i unwind label %lpad.i.i.i27.i.i

lpad.i.i.i27.i.i:		; preds = %_ZN10FullMatrixIdEC1Ejj.exit.i.i
        %exn.i.i.i27.i.i = landingpad {ptr, i32}
                 cleanup
	invoke void @_Unwind_Resume(ptr undef)
			to label %.noexc.i.i unwind label %lpad.i.i

.noexc.i.i:		; preds = %lpad.i.i.i27.i.i
	unreachable

_ZN10FullMatrixIdEC1Ejj.exit28.i.i:		; preds = %_ZN10FullMatrixIdEC1Ejj.exit.i.i
	br i1 undef, label %bb58.i.i, label %bb.i.i.i304.i.i

bb.i.i.i304.i.i:		; preds = %_ZN10FullMatrixIdEC1Ejj.exit28.i.i
	unreachable

bb58.i.i:		; preds = %_ZN10FullMatrixIdEC1Ejj.exit28.i.i
	br i1 false, label %bb.i191.i, label %bb.i.i.i297.i.i

bb.i.i.i297.i.i:		; preds = %bb58.i.i
	unreachable

lpad.i.i:		; preds = %lpad.i.i.i27.i.i
        %exn.i.i = landingpad {ptr, i32}
                 cleanup
	unreachable

bb.i191.i:		; preds = %.noexc232.i, %bb58.i.i
	invoke fastcc void @_ZN9TableBaseILi2EdE6reinitERK12TableIndicesILi2EE(ptr undef, i32 undef, i32 undef)
			to label %.noexc232.i unwind label %lpad196.i

.noexc232.i:		; preds = %bb.i191.i
	br i1 undef, label %bb29.loopexit.i.i, label %bb.i191.i

bb7.i215.i:		; preds = %bb9.i216.i
	br i1 undef, label %bb16.preheader.i.i, label %bb8.i.i

bb8.i.i:		; preds = %bb7.i215.i
	%tmp60.i.i = add i32 %0, 1		; <i32> [#uses=1]
	br label %bb9.i216.i

bb9.i216.i:		; preds = %bb29.loopexit.i.i, %bb8.i.i
	%0 = phi i32 [ 0, %bb29.loopexit.i.i ], [ %tmp60.i.i, %bb8.i.i ]		; <i32> [#uses=2]
	br i1 undef, label %bb7.i215.i, label %bb16.preheader.i.i

bb15.i.i:		; preds = %bb16.preheader.i.i, %bb15.i.i
	%j1.0212.i.i = phi i32 [ %1, %bb15.i.i ], [ 0, %bb16.preheader.i.i ]		; <i32> [#uses=2]
	%tmp6.i.i195.i.i = load i32, ptr undef, align 4		; <i32> [#uses=1]
	%tmp231.i.i = mul i32 %0, %tmp6.i.i195.i.i		; <i32> [#uses=1]
	%tmp13.i197.i.i = add i32 %j1.0212.i.i, %tmp231.i.i		; <i32> [#uses=0]
	%1 = add i32 %j1.0212.i.i, 1		; <i32> [#uses=1]
	br i1 undef, label %bb15.i.i, label %bb17.i.i

bb17.i.i:		; preds = %bb16.preheader.i.i, %bb15.i.i
	br label %bb16.preheader.i.i

bb16.preheader.i.i:		; preds = %bb17.i.i, %bb9.i216.i, %bb7.i215.i
	br i1 undef, label %bb17.i.i, label %bb15.i.i

bb29.loopexit.i.i:		; preds = %.noexc232.i
	br label %bb9.i216.i

lpad.i:		; preds = %entry
        %exn.i = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad120.i:		; preds = %invcont.i
        %exn120.i = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad124.i:		; preds = %invcont1.i
        %exn124.i = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad128.i:		; preds = %invcont3.i
        %exn128.i = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad132.i:		; preds = %invcont4.i
        %exn132.i = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad136.i:		; preds = %invcont6.i
        %exn136.i = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad140.i:		; preds = %bb21.i, %invcont7.i
        %exn140.i = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad144.i:		; preds = %bb10.i168.i, %invcont9.i
        %exn144.i = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad148.i:		; preds = %invcont10.i
        %exn148.i = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad188.i:		; preds = %bb50.i.i.i
        %exn188.i = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad196.i:		; preds = %bb.i191.i
        %exn196 = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad200.i:		; preds = %bb50.i.i
        %exn200.i = landingpad {ptr, i32}
                 cleanup
	unreachable

lpad204.i:		; preds = %invcont86.i
        %exn204.i = landingpad {ptr, i32}
                 cleanup
	unreachable
}

declare fastcc void @_ZN11Polynomials19LagrangeEquidistant23generate_complete_basisEj(ptr noalias nocapture sret(%"struct.std::vector<Polynomials::Polynomial<double>,std::allocator<Polynomials::Polynomial<double> > >"), i32)

declare i32 @__gxx_personality_v0(...)
