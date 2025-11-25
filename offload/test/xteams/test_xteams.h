
/*=============================== test_xteams.h -=============================//


Headerfile for testing the Cross-Team Scan Implementation in the DeviceRTL.
Also contains headers for the kmpc_ functions defined in the DeviceRTL/src/
Xteams.cpp.

//===----------------------------------------------------------------------===*/

#include "../xteamr/test_xteamr.h"  // include reduction helper functions rfun_*

#define _CD double _Complex
#define _CF float _Complex
#define _UI unsigned int
#define _UL unsigned long
#define _INLINE_ATTR_ __attribute__((flatten, always_inline))

// Headers for extern xteams functions defined in libomptarget DeviceRTL
// are defined here in the test header file because user apps cannot include
// the DeviceRTL Xteams.h header file.

#if defined(__AMDGCN__) || defined(__NVPTX__)
extern "C" {
#define _RF_LDS volatile __attribute__((address_space(3)))
void _INLINE_ATTR_  __kmpc_xteams_d_16x64
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_f_16x64
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cd_16x64
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cf_16x64
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_i_16x64
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ui_16x64
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_l_16x64
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ul_16x64
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_d_8x64
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_f_8x64
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cd_8x64
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cf_8x64
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_i_8x64
   (int v, int* storage, int* r_array, int* tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ui_8x64
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_l_8x64
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ul_8x64
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_d_4x64
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_f_4x64
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cd_4x64
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cf_4x64
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_i_4x64
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ui_4x64
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_l_4x64
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ul_4x64
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_d_2x64
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_f_2x64
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cd_2x64
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cf_2x64
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_i_2x64
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ui_2x64
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_l_2x64
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ul_2x64
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_d_1x64
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_f_1x64
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cd_1x64
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cf_1x64
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_i_1x64
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ui_1x64
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_l_1x64
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ul_1x64
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_d_32x32
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_f_32x32
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cd_32x32
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cf_32x32
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_i_32x32
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ui_32x32
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_l_32x32
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ul_32x32
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_d_16x32
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_f_16x32
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cd_16x32
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cf_16x32
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_i_16x32
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ui_16x32
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_l_16x32
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ul_16x32
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_d_8x32
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_f_8x32
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cd_8x32
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cf_8x32
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_i_8x32
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ui_8x32
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_l_8x32
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ul_8x32
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_d_4x32
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_f_4x32
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cd_4x32
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cf_4x32
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_i_4x32
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ui_4x32
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_l_4x32
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ul_4x32
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_d_2x32
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_f_2x32
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cd_2x32
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_cf_2x32
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_i_2x32
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ui_2x32
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_l_2x32
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams);
void _INLINE_ATTR_  __kmpc_xteams_ul_2x32
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams);
} // end extern C

#else

// For host compilation, define null functions for host linking.

extern "C" {
#undef _RF_LDS
#define _RF_LDS
void  __kmpc_xteams_d_16x64
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_f_16x64
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cd_16x64
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cf_16x64
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_i_16x64
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ui_16x64
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_l_16x64
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ul_16x64
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_d_8x64
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_f_8x64
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cd_8x64
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cf_8x64
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_i_8x64
   (int v, int* storage, int* r_array, int* tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ui_8x64
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_l_8x64
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ul_8x64
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_d_4x64
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_f_4x64
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cd_4x64
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cf_4x64
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_i_4x64
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ui_4x64
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_l_4x64
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ul_4x64
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_d_2x64
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_f_2x64
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cd_2x64
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cf_2x64
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_i_2x64
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ui_2x64
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_l_2x64
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ul_2x64
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_d_1x64
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_f_1x64
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cd_1x64
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cf_1x64
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_i_1x64
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ui_1x64
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_l_1x64
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ul_1x64
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_d_32x32
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_f_32x32
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cd_32x32
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cf_32x32
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_i_32x32
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ui_32x32
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_l_32x32
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ul_32x32
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_d_16x32
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_f_16x32
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cd_16x32
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cf_16x32
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_i_16x32
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ui_16x32
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_l_16x32
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ul_16x32
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_d_8x32
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_f_8x32
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cd_8x32
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cf_8x32
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_i_8x32
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ui_8x32
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_l_8x32
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ul_8x32
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_d_4x32
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_f_4x32
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cd_4x32
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cf_4x32
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_i_4x32
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ui_4x32
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_l_4x32
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ul_4x32
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_d_2x32
   (double v, double* storage, double* r_array, double *tvs, uint32_t *td, void (*_rf)(double *, double),
      void (*_rf_lds)(_RF_LDS double *, _RF_LDS double *), const double iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_f_2x32
   (float v, float* storage, float* r_array, float *tvs, uint32_t *td, void (*_rf)(float *, float),
      void (*_rf_lds)(_RF_LDS float *, _RF_LDS float *), const float iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cd_2x32
   (_CD v, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, void (*_rf)(_CD *, _CD),
      void (*_rf_lds)(_RF_LDS _CD *, _RF_LDS _CD *), const _CD iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_cf_2x32
   (_CF v, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, void (*_rf)(_CF *, _CF),
      void (*_rf_lds)(_RF_LDS _CF *, _RF_LDS _CF *), const _CF iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_i_2x32
   (int v, int* storage, int* r_array, int *tvs, uint32_t *td, void (*_rf)(int *, int),
      void (*_rf_lds)(_RF_LDS int *, _RF_LDS int *), const int iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ui_2x32
   (_UI v, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, void (*_rf)(_UI *, _UI),
      void (*_rf_lds)(_RF_LDS _UI *, _RF_LDS _UI *), const _UI iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_l_2x32
   (long v, long* storage, long* r_array, long *tvs, uint32_t *td, void (*_rf)(long *, long),
      void (*_rf_lds)(_RF_LDS long *, _RF_LDS long *), const long iv,
      const uint64_t k, const uint32_t numteams){};
void  __kmpc_xteams_ul_2x32
   (_UL v, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, void (*_rf)(_UL *, _UL),
      void (*_rf_lds)(_RF_LDS _UL *, _RF_LDS _UL *), const _UL iv,
      const uint64_t k, const uint32_t numteams){};
} // end extern C

#endif  // of definitions for host null functions

// These overloaded function definitions are for this test framework 
// (test_xteams.cpp) to invoke the extern DeviceRTL helper functions.

void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_d, __kmpc_rfun_sum_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_f, __kmpc_rfun_sum_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x64
   (_CD val, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, const _CD iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cd_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cd, __kmpc_rfun_sum_lds_cd, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x64
   (_CF val, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, const _CF iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cf_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cf, __kmpc_rfun_sum_lds_cf, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ui, __kmpc_rfun_sum_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_l, __kmpc_rfun_sum_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ul, __kmpc_rfun_sum_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_d, __kmpc_rfun_sum_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_f, __kmpc_rfun_sum_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x64
   (_CD val, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, const _CD iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cd_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cd, __kmpc_rfun_sum_lds_cd, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x64
   (_CF val, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, const _CF iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cf_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cf, __kmpc_rfun_sum_lds_cf, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, iv, k, numteams);}
   // (int val, int* storage, int* r_array, void* lc0_struct, const uint64_t k, const uint32_t numteams)
   // { __kmpc_xteams_i_8x64(val, storage, r_array, lc0_struct,
   //    __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ui, __kmpc_rfun_sum_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_l, __kmpc_rfun_sum_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ul, __kmpc_rfun_sum_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_d, __kmpc_rfun_sum_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_f, __kmpc_rfun_sum_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x64
   (_CD val, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, const _CD iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cd_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cd, __kmpc_rfun_sum_lds_cd, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x64
   (_CF val, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, const _CF iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cf_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cf, __kmpc_rfun_sum_lds_cf, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ui, __kmpc_rfun_sum_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_l, __kmpc_rfun_sum_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ul, __kmpc_rfun_sum_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_d, __kmpc_rfun_sum_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_f, __kmpc_rfun_sum_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x64
   (_CD val, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, const _CD iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cd_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cd, __kmpc_rfun_sum_lds_cd, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x64
   (_CF val, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, const _CF iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cf_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cf, __kmpc_rfun_sum_lds_cf, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ui, __kmpc_rfun_sum_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_l, __kmpc_rfun_sum_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ul, __kmpc_rfun_sum_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_1x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_d, __kmpc_rfun_sum_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_1x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_f, __kmpc_rfun_sum_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_1x64
   (_CD val, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, const _CD iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cd_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cd, __kmpc_rfun_sum_lds_cd, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_1x64
   (_CF val, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, const _CF iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cf_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cf, __kmpc_rfun_sum_lds_cf, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_1x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_1x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ui, __kmpc_rfun_sum_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_1x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_l, __kmpc_rfun_sum_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_1x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ul, __kmpc_rfun_sum_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_32x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_d, __kmpc_rfun_sum_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_32x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_f, __kmpc_rfun_sum_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_32x32
   (_CD val, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, const _CD iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cd_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cd, __kmpc_rfun_sum_lds_cd, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_32x32
   (_CF val, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, const _CF iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cf_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cf, __kmpc_rfun_sum_lds_cf, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_32x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_32x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ui, __kmpc_rfun_sum_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_32x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_l, __kmpc_rfun_sum_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_32x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ul, __kmpc_rfun_sum_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_d, __kmpc_rfun_sum_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_f, __kmpc_rfun_sum_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x32
   (_CD val, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, const _CD iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cd_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cd, __kmpc_rfun_sum_lds_cd, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x32
   (_CF val, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, const _CF iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cf_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cf, __kmpc_rfun_sum_lds_cf, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ui, __kmpc_rfun_sum_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_l, __kmpc_rfun_sum_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_16x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ul, __kmpc_rfun_sum_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_d, __kmpc_rfun_sum_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_f, __kmpc_rfun_sum_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x32
   (_CD val, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, const _CD iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cd_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cd, __kmpc_rfun_sum_lds_cd, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x32
   (_CF val, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, const _CF iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cf_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cf, __kmpc_rfun_sum_lds_cf, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ui, __kmpc_rfun_sum_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_l, __kmpc_rfun_sum_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_8x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ul, __kmpc_rfun_sum_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_d, __kmpc_rfun_sum_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_f, __kmpc_rfun_sum_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x32
   (_CD val, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, const _CD iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cd_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cd, __kmpc_rfun_sum_lds_cd, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x32
   (_CF val, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, const _CF iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cf_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cf, __kmpc_rfun_sum_lds_cf, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ui, __kmpc_rfun_sum_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_l, __kmpc_rfun_sum_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_4x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ul, __kmpc_rfun_sum_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_d, __kmpc_rfun_sum_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_f, __kmpc_rfun_sum_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x32
   (_CD val, _CD* storage, _CD* r_array, _CD *tvs, uint32_t *td, const _CD iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cd_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cd, __kmpc_rfun_sum_lds_cd, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x32
   (_CF val, _CF* storage, _CF* r_array, _CF *tvs, uint32_t *td, const _CF iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_cf_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_cf, __kmpc_rfun_sum_lds_cf, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_i, __kmpc_rfun_sum_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ui, __kmpc_rfun_sum_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_l, __kmpc_rfun_sum_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_sum_2x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_sum_ul, __kmpc_rfun_sum_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_d, __kmpc_rfun_max_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_f, __kmpc_rfun_max_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_i, __kmpc_rfun_max_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ui, __kmpc_rfun_max_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_l, __kmpc_rfun_max_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ul, __kmpc_rfun_max_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_d, __kmpc_rfun_max_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_f, __kmpc_rfun_max_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_i, __kmpc_rfun_max_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ui, __kmpc_rfun_max_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_l, __kmpc_rfun_max_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ul, __kmpc_rfun_max_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_d, __kmpc_rfun_max_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_f, __kmpc_rfun_max_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_i, __kmpc_rfun_max_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ui, __kmpc_rfun_max_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_l, __kmpc_rfun_max_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ul, __kmpc_rfun_max_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_d, __kmpc_rfun_max_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_f, __kmpc_rfun_max_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_i, __kmpc_rfun_max_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ui, __kmpc_rfun_max_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_l, __kmpc_rfun_max_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ul, __kmpc_rfun_max_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_1x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_d, __kmpc_rfun_max_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_1x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_f, __kmpc_rfun_max_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_1x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_i, __kmpc_rfun_max_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_1x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ui, __kmpc_rfun_max_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_1x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_l, __kmpc_rfun_max_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_1x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ul, __kmpc_rfun_max_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_32x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_d, __kmpc_rfun_max_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_32x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_f, __kmpc_rfun_max_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_32x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_i, __kmpc_rfun_max_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_32x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ui, __kmpc_rfun_max_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_32x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_l, __kmpc_rfun_max_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_32x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ul, __kmpc_rfun_max_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_d, __kmpc_rfun_max_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_f, __kmpc_rfun_max_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_i, __kmpc_rfun_max_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ui, __kmpc_rfun_max_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_l, __kmpc_rfun_max_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_16x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ul, __kmpc_rfun_max_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_d, __kmpc_rfun_max_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_f, __kmpc_rfun_max_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_i, __kmpc_rfun_max_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ui, __kmpc_rfun_max_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_l, __kmpc_rfun_max_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_8x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ul, __kmpc_rfun_max_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_d, __kmpc_rfun_max_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_f, __kmpc_rfun_max_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_i, __kmpc_rfun_max_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ui, __kmpc_rfun_max_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_l, __kmpc_rfun_max_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_4x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ul, __kmpc_rfun_max_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_d, __kmpc_rfun_max_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_f, __kmpc_rfun_max_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_i, __kmpc_rfun_max_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ui, __kmpc_rfun_max_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_l, __kmpc_rfun_max_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_max_2x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_max_ul, __kmpc_rfun_max_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_d, __kmpc_rfun_min_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_f, __kmpc_rfun_min_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_i, __kmpc_rfun_min_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ui, __kmpc_rfun_min_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_l, __kmpc_rfun_min_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_16x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ul, __kmpc_rfun_min_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_d, __kmpc_rfun_min_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_f, __kmpc_rfun_min_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_i, __kmpc_rfun_min_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ui, __kmpc_rfun_min_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_l, __kmpc_rfun_min_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_8x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ul, __kmpc_rfun_min_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_d, __kmpc_rfun_min_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_f, __kmpc_rfun_min_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_i, __kmpc_rfun_min_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ui, __kmpc_rfun_min_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_l, __kmpc_rfun_min_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_4x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ul, __kmpc_rfun_min_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_d, __kmpc_rfun_min_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_f, __kmpc_rfun_min_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_i, __kmpc_rfun_min_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ui, __kmpc_rfun_min_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_l, __kmpc_rfun_min_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_2x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ul, __kmpc_rfun_min_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_1x64
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_d, __kmpc_rfun_min_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_1x64
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_f, __kmpc_rfun_min_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_1x64
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_i, __kmpc_rfun_min_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_1x64
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ui, __kmpc_rfun_min_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_1x64
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_l, __kmpc_rfun_min_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_1x64
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_1x64(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ul, __kmpc_rfun_min_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_32x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_d, __kmpc_rfun_min_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_32x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_f, __kmpc_rfun_min_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_32x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_i, __kmpc_rfun_min_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_32x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ui, __kmpc_rfun_min_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_32x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_l, __kmpc_rfun_min_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_32x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_32x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ul, __kmpc_rfun_min_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_d, __kmpc_rfun_min_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_f, __kmpc_rfun_min_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_i, __kmpc_rfun_min_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ui, __kmpc_rfun_min_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_l, __kmpc_rfun_min_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_16x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_16x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ul, __kmpc_rfun_min_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_d, __kmpc_rfun_min_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_f, __kmpc_rfun_min_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_i, __kmpc_rfun_min_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ui, __kmpc_rfun_min_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_l, __kmpc_rfun_min_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_8x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_8x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ul, __kmpc_rfun_min_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_d, __kmpc_rfun_min_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_f, __kmpc_rfun_min_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_i, __kmpc_rfun_min_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ui, __kmpc_rfun_min_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_l, __kmpc_rfun_min_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_4x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_4x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ul, __kmpc_rfun_min_lds_ul, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x32
   (double val, double* storage, double* r_array, double *tvs, uint32_t *td, const double iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_d_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_d, __kmpc_rfun_min_lds_d, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x32
   (float val, float* storage, float* r_array, float *tvs, uint32_t *td, const float iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_f_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_f, __kmpc_rfun_min_lds_f, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x32
   (int val, int* storage, int* r_array, int *tvs, uint32_t *td, const int iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_i_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_i, __kmpc_rfun_min_lds_i, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x32
   (_UI val, _UI* storage, _UI* r_array, _UI *tvs, uint32_t *td, const _UI iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ui_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ui, __kmpc_rfun_min_lds_ui, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x32
   (long val, long* storage, long* r_array, long *tvs, uint32_t *td, const long iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_l_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_l, __kmpc_rfun_min_lds_l, iv, k, numteams);}
void _INLINE_ATTR_ _overload_to_extern_scan_min_2x32
   (_UL val, _UL* storage, _UL* r_array, _UL *tvs, uint32_t *td, const _UL iv, const uint64_t k, const uint32_t numteams)
   { __kmpc_xteams_ul_2x32(val, storage, r_array, tvs, td,
      __kmpc_rfun_min_ul, __kmpc_rfun_min_lds_ul, iv, k, numteams);}
#undef _CD
#undef _CF
#undef _UI
#undef _UL
#undef _INLINE_ATTR_
