
;-------------------------------------------------------------------------------
; NaN tests.
;-------------------------------------------------------------------------------

fadd_dyn__nan_pnorm__nan_pnorm__ignore
fadd_dyn__nan_pnorm__nan_pnorm__maytrap
fadd_dyn__nan_pnorm__nan_pnorm__strict

fsub_dyn__nan_pnorm__nan_pnorm__ignore
fsub_dyn__nan_pnorm__nan_pnorm__maytrap
fsub_dyn__nan_pnorm__nan_pnorm__strict

fadd_dyn__qnan_pnorm__qnan_pnorm__ignore
fadd_dyn__qnan_pnorm__qnan_pnorm__maytrap
fadd_dyn__qnan_pnorm__qnan_pnorm__strict

fadd_dyn__snan_pnorm__snan_pnorm__ignore
fadd_dyn__snan_pnorm__snan_pnorm__maytrap
fadd_dyn__snan_pnorm__snan_pnorm__strict

fadd_dyn__inf__inf__ignore
fadd_dyn__inf__inf__maytrap
fadd_dyn__inf__inf__strict

;-------------------------------------------------------------------------------
; Test denormal modes.
;-------------------------------------------------------------------------------

define float @fadd_rtn__nsub__pnorm__dynamic_dynamic
define float @fadd_rtn__nsub__pnorm__ftz_daz
define float @fadd_rtn__nsub__pnorm__ftpz_dapz

define float @fsub_rtn__nsub__pnorm__dynamic_dynamic
define float @fsub_rtn__nsub__pnorm__ftz_daz
define float @fsub_rtn__nsub__pnorm__ftpz_dapz

define float @fsub_rtp__nzero__psub__ieee_dynamic
define float @fsub_rtp__nzero__psub__ieee_daz
define float @fsub_rtp__nzero__psub__ieee_dapz

define float @fsub_rtn__pzero__psub__ieee_dynamic
define float @fsub_rtn__pzero__psub__ieee_daz
define float @fsub_rtn__pzero__psub__ieee_dapz

;-------------------------------------------------------------------------------
; fadd_self
;-------------------------------------------------------------------------------

fadd_self_dyn__norm__maybe_undef

fadd_self_dyn__norm
fadd_self_rne__norm
fadd_self_rna__norm
fadd_self_rtz__norm
fadd_self_rtp__norm
fadd_self_rtn__norm

fadd_self_dyn__nan_pnorm__ignore
fadd_self_dyn__nan_pnorm__maytrap
fadd_self_dyn__nan_pnorm__strict

fadd_self_dyn__nsub__dynamic_dynamic
fadd_self_dyn__nsub__ftpz_dapz

;-------------------------------------------------------------------------------
; fsub_self
;-------------------------------------------------------------------------------

fsub_self_dyn__norm__maybe_undef

fsub_self_dyn__norm
fsub_self_rne__norm
fsub_self_rna__norm
fsub_self_rtz__norm
fsub_self_rtp__norm
fsub_self_rtn__norm

fsub_self_dyn__nan_pnorm__ignore
fsub_self_dyn__nan_pnorm__maytrap
fsub_self_dyn__nan_pnorm__strict

fsub_self_rtp__nsub__dynamic_dynamic
fsub_self_rtp__nsub__ftpz_dapz

fsub_self_rtn__nsub__dynamic_dynamic
fsub_self_rtn__nsub__ftpz_dapz

;-------------------------------------------------------------------------------
; Zero tests.
;-------------------------------------------------------------------------------

fadd_dyn__qnan_pzero__qnan_nzero
fadd_rtp__qnan_pzero__qnan_nzero
fadd_rtn__qnan_pzero__qnan_nzero

fadd_dyn__qnan_nzero__qnan_pzero
fadd_rtp__qnan_nzero__qnan_pzero
fadd_rtn__qnan_nzero__qnan_pzero

fsub_dyn__qnan_pzero__qnan_pzero
fsub_rtp__qnan_pzero__qnan_pzero
fsub_rtn__qnan_pzero__qnan_pzero

fsub_dyn__qnan_nzero__qnan_nzero
fsub_rtp__qnan_nzero__qnan_nzero
fsub_rtn__qnan_nzero__qnan_nzero

;-------------------------------------------------------------------------------
; Overflow tests.
;-------------------------------------------------------------------------------

fadd_dyn__pnorm__psub
fadd_dyn__psub__pnorm
fadd_rtp__pnorm__psub
fadd_rtp__psub__pnorm

fsub_dyn__pnorm__nsub
fsub_dyn__psub__nnorm
fsub_rtp__pnorm__nsub
fsub_rtp__psub__nnorm

fadd_dyn__nnorm__nsub
fadd_dyn__nsub__nnorm
fadd_rtn__nnorm__nsub
fadd_rtn__nsub__nnorm

fsub_dyn__nnorm__psub
fsub_dyn__nsub__pnorm
fsub_rtp__nnorm__psub
fsub_rtp__nsub__pnorm

;-------------------------------------------------------------------------------

fadd_rtz__norm__norm
fsub_rtz__norm__norm

fadd_rtz__pinf_norm__norm
fsub_rtz__norm__ninf_norm

fadd_dyn__pnorm__pnorm
fadd_rtn__pnorm__pnorm

fsub_dyn__pnorm__nnorm
fsub_rtn__pnorm__nnorm

fadd_dyn__nnorm__nnorm
fadd_rtp__nnorm__nnorm

fsub_dyn__nnorm__pnorm
fsub_rtp__nnorm__pnorm

fadd_rna__norm__sub
fsub_rna__sub__norm

;-------------------------------------------------------------------------------
