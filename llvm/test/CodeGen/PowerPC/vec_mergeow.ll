; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | \
; RUN:   FileCheck %s  -check-prefix=CHECK-LE
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | \
; RUN:   FileCheck %s -check-prefix=CHECK-BE

; Check for a vector merge instruction using two inputs
; The shufflevector specifies the even elements, using big endian element 
; ordering. If run on a big endian machine, this should produce the vmrgew 
; instruction. If run on a little endian machine, this should produce the
; vmrgow instruction. Note also that on little endian the input registers 
; are swapped also.
define void @check_merge_even_xy(<16 x i8>* %A, <16 x i8>* %B) {
entry:
; CHECK-LE-LABEL: @check_merge_even_xy
; CHECK-BE-LABEL: @check_merge_even_xy
        %tmp = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp2, 
	      		      <16 x i32> <i32 0, i32 1, i32 2, i32 3, 
			      	    	  i32 16, i32 17, i32 18, i32 19, 
					  i32 8, i32 9, i32 10, i32 11, 
					  i32 24, i32 25, i32 26, i32 27>
; CHECK-LE: vmrgow 2, 3, 2
; CHECK-BE: vmrgew 2, 2, 3
      	store <16 x i8> %tmp3, <16 x i8>* %A
	ret void
; CHECK-LE: blr
; CHECK-BE: blr
}

; Check for a vector merge instruction using a single input. 
; The shufflevector specifies the even elements, using big endian element 
; ordering. If run on a big endian machine, this should produce the vmrgew 
; instruction. If run on a little endian machine, this should produce the
; vmrgow instruction. 
define void @check_merge_even_xx(<16 x i8>* %A) {
entry:
; CHECK-LE-LABEL: @check_merge_even_xx
; CHECK-BE-LABEL: @check_merge_even_xx
        %tmp = load <16 x i8>, <16 x i8>* %A
	%tmp2 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp, 
	      		      <16 x i32> <i32 0, i32 1, i32 2, i32 3, 
			      	          i32 0, i32 1, i32 2, i32 3, 
					  i32 8, i32 9, i32 10, i32 11, 
					  i32 8, i32 9, i32 10, i32 11>
; CHECK-LE: vmrgow 2, 2, 2
; CHECK-BE: vmrgew 2, 2, 2
  	store <16 x i8> %tmp2, <16 x i8>* %A
	ret void
; CHECK-LE: blr
; CHECK-BE: blr       
}

; Check for a vector merge instruction using two inputs.
; The shufflevector specifies the odd elements, using big endian element 
; ordering. If run on a big endian machine, this should produce the vmrgow 
; instruction. If run on a little endian machine, this should produce the
; vmrgew instruction. Note also that on little endian the input registers 
; are swapped also.
define void @check_merge_odd_xy(<16 x i8>* %A, <16 x i8>* %B) {
entry:
; CHECK-LE-LABEL: @check_merge_odd_xy
; CHECK-BE-LABEL: @check_merge_odd_xy
        %tmp = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp2, 
	      		      <16 x i32> <i32 4, i32 5, i32 6, i32 7, 
			      	    	  i32 20, i32 21, i32 22, i32 23, 
					  i32 12, i32 13, i32 14, i32 15, 
					  i32 28, i32 29, i32 30, i32 31>
; CHECK-LE: vmrgew 2, 3, 2
; CHECK-BE: vmrgow 2, 2, 3
        store <16 x i8> %tmp3, <16 x i8>* %A
	ret void
; CHECK-LE: blr
; CHECK-BE: blr
}

; Check for a vector merge instruction using a single input.
; The shufflevector specifies the odd elements, using big endian element 
; ordering. If run on a big endian machine, this should produce the vmrgow 
; instruction. If run on a little endian machine, this should produce the
; vmrgew instruction. 
define void @check_merge_odd_xx(<16 x i8>* %A) {
entry:
; CHECK-LE-LABEL: @check_merge_odd_xx
; CHECK-BE-LABEL: @check_merge_odd_xx
        %tmp = load <16 x i8>, <16 x i8>* %A
	%tmp2 = shufflevector <16 x i8> %tmp, <16 x i8> %tmp, 
	      		      <16 x i32> <i32 4, i32 5, i32 6, i32 7, 
			      	    	  i32 4, i32 5, i32 6, i32 7, 
					  i32 12, i32 13, i32 14, i32 15, 
					  i32 12, i32 13, i32 14, i32 15>
; CHECK-LE: vmrgew 2, 2, 2
; CHECK-BE: vmrgow 2, 2, 2
        store <16 x i8> %tmp2, <16 x i8>* %A
	ret void
; CHECK-LE: blr
; CHECK-BE: blr
}

