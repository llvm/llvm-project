__attribute__((overloadable, always_inline)) int __spirv_ImageQuerySize(image1d_buffer_t img);
__attribute__((overloadable, always_inline)) int2 __spirv_ImageQuerySize(image1d_array_t img);
__attribute__((overloadable, always_inline)) int2 __spirv_ImageQuerySize(image2d_t img);
__attribute__((overloadable, always_inline)) int2 __spirv_ImageQuerySize(image2d_depth_t img);
__attribute__((overloadable, always_inline)) int3 __spirv_ImageQuerySize(image2d_array_t img);
__attribute__((overloadable, always_inline)) int3 __spirv_ImageQuerySize(image2d_array_depth_t img);

__attribute__((overloadable, always_inline)) int __spirv_ImageQuerySizeLod(image1d_t img, int lod);
__attribute__((overloadable, always_inline)) int2 __spirv_ImageQuerySizeLod(image1d_array_t img, int lod);
__attribute__((overloadable, always_inline)) int2 __spirv_ImageQuerySizeLod(image2d_t img, int lod);
__attribute__((overloadable, always_inline)) int2 __spirv_ImageQuerySizeLod(image2d_depth_t img, int lod);
__attribute__((overloadable, always_inline)) int3 __spirv_ImageQuerySizeLod(image2d_array_t img, int lod);
__attribute__((overloadable, always_inline)) int3 __spirv_ImageQuerySizeLod(image2d_array_depth_t img, int lod);
__attribute__((overloadable, always_inline)) int3 __spirv_ImageQuerySizeLod(image3d_t img, int lod);

