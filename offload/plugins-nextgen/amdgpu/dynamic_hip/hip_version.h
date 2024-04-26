#ifndef HIP_VERSION
#define HIP_VERSION

#define HIP_VERSION_MAJOR 0
#define HIP_VERSION_MINOR 0
#define HIP_VERSION_PATCH 0
#define HIP_VERSION_GITHASH ""
#define HIP_VERSION_BUILD_ID 0
#define HIP_VERSION_BUILD_NAME ""
#define HIP_VERSION                                                            \
  (HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 +                 \
   HIP_VERSION_PATCH)

#define __HIP_HAS_GET_PCH 0

#endif /* HIP_VERSION */
