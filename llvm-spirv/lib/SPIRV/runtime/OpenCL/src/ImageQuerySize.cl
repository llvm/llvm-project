#include "../inc/spirv.h"

__attribute__((overloadable, always_inline)) int  __spirv_ImageQuerySize(image1d_buffer_t img) {
  return get_image_width(img);
}

__attribute__((overloadable, always_inline)) int  __spirv_ImageQuerySizeLod(image1d_t img, int lod) {
  return get_image_width(img) >> lod;
}
__attribute__((overloadable, always_inline)) int2  __spirv_ImageQuerySize(image1d_array_t img) {
  return (int2)(get_image_width(img), get_image_array_size(img));
}
__attribute__((overloadable, always_inline)) int2  __spirv_ImageQuerySizeLod(image1d_array_t img, int lod) {
  return (int2)(get_image_width(img) >> lod, get_image_array_size(img) >> lod);
}

#define DEFINE_SPIRV_ImageQuerySizeLod_2d(ImgTy) \
__attribute__((overloadable, always_inline)) int2 __spirv_ImageQuerySizeLod(ImgTy img, int lod) { \
  return get_image_dim(img) >> lod; \
}

#define DEFINE_SPIRV_ImageQuerySizeLod_2darray(ImgTy) \
__attribute__((overloadable, always_inline)) int3 __spirv_ImageQuerySizeLod(ImgTy img, int lod) { \
  return (int3)(get_image_dim(img) >> lod, get_image_array_size(img) >> lod); \
}

#define DEFINE_SPIRV_ImageQuerySize_2d(ImgTy) \
__attribute__((overloadable, always_inline)) int2 __spirv_ImageQuerySize(ImgTy img) { \
  return get_image_dim(img); \
}

#define DEFINE_SPIRV_ImageQuerySize_2darray(ImgTy) \
__attribute__((overloadable, always_inline)) int3 __spirv_ImageQuerySize(ImgTy img) { \
  return (int3)(get_image_dim(img), get_image_array_size(img)); \
}

__attribute__((overloadable, always_inline)) int3 __spirv_ImageQuerySizeLod(image3d_t img, int lod) {
  return get_image_dim(img).xyz >> lod;
}

DEFINE_SPIRV_ImageQuerySize_2d(image2d_t)
DEFINE_SPIRV_ImageQuerySize_2d(image2d_depth_t)
DEFINE_SPIRV_ImageQuerySizeLod_2d(image2d_t)
DEFINE_SPIRV_ImageQuerySizeLod_2d(image2d_depth_t)
DEFINE_SPIRV_ImageQuerySize_2darray(image2d_array_t)
DEFINE_SPIRV_ImageQuerySize_2darray(image2d_array_depth_t)
DEFINE_SPIRV_ImageQuerySizeLod_2darray(image2d_array_t)
DEFINE_SPIRV_ImageQuerySizeLod_2darray(image2d_array_depth_t)
