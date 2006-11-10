; RUN: llvm-as < %s | llc -march=x86 | grep 'subl $4, %esp'

%str = internal constant [9 x sbyte] c"%f+%f*i\0A\00"		; <[9 x sbyte]*> [#uses=1]

implementation   ; Functions:

int %main() {
entry:
	%retval = alloca int, align 4		; <int*> [#uses=1]
	%tmp = alloca { double, double }, align 16		; <{ double, double }*> [#uses=4]
	%tmp1 = alloca { double, double }, align 16		; <{ double, double }*> [#uses=4]
	%tmp2 = alloca { double, double }, align 16		; <{ double, double }*> [#uses=3]
	%pi = alloca double, align 8		; <double*> [#uses=2]
	%z = alloca { double, double }, align 16		; <{ double, double }*> [#uses=4]
	"alloca point" = cast int 0 to int		; <int> [#uses=0]
	store double 0x400921FB54442D18, double* %pi
	%tmp = load double* %pi		; <double> [#uses=1]
	%real = getelementptr { double, double }* %tmp1, uint 0, uint 0		; <double*> [#uses=1]
	store double 0.000000e+00, double* %real
	%real3 = getelementptr { double, double }* %tmp1, uint 0, uint 1		; <double*> [#uses=1]
	store double %tmp, double* %real3
	%tmp = getelementptr { double, double }* %tmp, uint 0, uint 0		; <double*> [#uses=1]
	%tmp4 = getelementptr { double, double }* %tmp1, uint 0, uint 0		; <double*> [#uses=1]
	%tmp5 = load double* %tmp4		; <double> [#uses=1]
	store double %tmp5, double* %tmp
	%tmp6 = getelementptr { double, double }* %tmp, uint 0, uint 1		; <double*> [#uses=1]
	%tmp7 = getelementptr { double, double }* %tmp1, uint 0, uint 1		; <double*> [#uses=1]
	%tmp8 = load double* %tmp7		; <double> [#uses=1]
	store double %tmp8, double* %tmp6
	%tmp = cast { double, double }* %tmp to { long, long }*		; <{ long, long }*> [#uses=1]
	%tmp = getelementptr { long, long }* %tmp, uint 0, uint 0		; <long*> [#uses=1]
	%tmp = load long* %tmp		; <long> [#uses=1]
	%tmp9 = cast { double, double }* %tmp to { long, long }*		; <{ long, long }*> [#uses=1]
	%tmp10 = getelementptr { long, long }* %tmp9, uint 0, uint 1		; <long*> [#uses=1]
	%tmp11 = load long* %tmp10		; <long> [#uses=1]
	call csretcc void %cexp( { double, double }* %tmp2, long %tmp, long %tmp11 )
	%tmp12 = getelementptr { double, double }* %z, uint 0, uint 0		; <double*> [#uses=1]
	%tmp13 = getelementptr { double, double }* %tmp2, uint 0, uint 0		; <double*> [#uses=1]
	%tmp14 = load double* %tmp13		; <double> [#uses=1]
	store double %tmp14, double* %tmp12
	%tmp15 = getelementptr { double, double }* %z, uint 0, uint 1		; <double*> [#uses=1]
	%tmp16 = getelementptr { double, double }* %tmp2, uint 0, uint 1		; <double*> [#uses=1]
	%tmp17 = load double* %tmp16		; <double> [#uses=1]
	store double %tmp17, double* %tmp15
	%tmp18 = getelementptr { double, double }* %z, uint 0, uint 1		; <double*> [#uses=1]
	%tmp19 = load double* %tmp18		; <double> [#uses=1]
	%tmp20 = getelementptr { double, double }* %z, uint 0, uint 0		; <double*> [#uses=1]
	%tmp21 = load double* %tmp20		; <double> [#uses=1]
	%tmp = getelementptr [9 x sbyte]* %str, int 0, uint 0		; <sbyte*> [#uses=1]
	%tmp = call int (sbyte*, ...)* %printf( sbyte* %tmp, double %tmp21, double %tmp19 )		; <int> [#uses=0]
	br label %return

return:		; preds = %entry
	%retval = load int* %retval		; <int> [#uses=1]
	ret int %retval
}

declare csretcc void %cexp({ double, double }*, long, long)

declare int %printf(sbyte*, ...)
