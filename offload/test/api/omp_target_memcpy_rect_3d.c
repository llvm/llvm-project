// RUN: %libomptarget-compile-and-run-generic | %fcheck-generic

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Reference implementation on host memory.
static void ref_rect(char *dst, const char *src, size_t es, int nd,
                     const size_t *vol, const size_t *doff, const size_t *soff,
                     const size_t *ddim, const size_t *sdim) {
  if (nd == 1) {
    memcpy(dst + doff[0] * es, src + soff[0] * es, vol[0] * es);
    return;
  }
  size_t dslice = es, sslice = es;
  for (int i = 1; i < nd; ++i) {
    dslice *= ddim[i];
    sslice *= sdim[i];
  }
  for (size_t i = 0; i < vol[0]; ++i)
    ref_rect(dst + (doff[0] + i) * dslice, src + (soff[0] + i) * sslice, es,
             nd - 1, vol + 1, doff + 1, soff + 1, ddim + 1, sdim + 1);
}

enum Dir { H2H, H2D, D2H, D2D };

static size_t total(int nd, const size_t *dim, size_t es) {
  size_t n = es;
  for (int i = 0; i < nd; ++i)
    n *= dim[i];
  return n;
}

// Run one copy in direction \p dir and compare against the reference.
static int run_case(const char *name, enum Dir dir, int nd, size_t es,
                    const size_t *vol, const size_t *doff, const size_t *soff,
                    const size_t *ddim, const size_t *sdim) {
  int host = omp_get_initial_device();
  int dev = omp_get_default_device();
  if (dev < 0 || dev >= omp_get_num_devices())
    dev = host;

  size_t sbytes = total(nd, sdim, es), dbytes = total(nd, ddim, es);
  char *hsrc = malloc(sbytes), *href = calloc(1, dbytes),
       *hout = calloc(1, dbytes);
  for (size_t i = 0; i < sbytes; ++i)
    hsrc[i] = (char)(i * 7 + 1);
  ref_rect(href, hsrc, es, nd, vol, doff, soff, ddim, sdim);

  int sdev = (dir == H2H || dir == H2D) ? host : dev;
  int ddev = (dir == H2H || dir == D2H) ? host : dev;
  char *src = sdev == host ? hsrc : omp_target_alloc(sbytes, sdev);
  char *dst = ddev == host ? hout : omp_target_alloc(dbytes, ddev);
  if (!src || !dst)
    return 1;
  if (sdev != host && omp_target_memcpy(src, hsrc, sbytes, 0, 0, sdev, host))
    return 1;
  if (ddev != host && omp_target_memcpy(dst, hout, dbytes, 0, 0, ddev, host))
    return 1;

  if (omp_target_memcpy_rect(dst, src, es, nd, vol, doff, soff, ddim, sdim,
                             ddev, sdev)) {
    printf("%s: rect copy failed\n", name);
    return 1;
  }
  if (ddev != host && omp_target_memcpy(hout, dst, dbytes, 0, 0, host, ddev))
    return 1;

  int rc = memcmp(href, hout, dbytes) != 0;
  if (rc)
    printf("%s: mismatch\n", name);
  if (sdev != host)
    omp_target_free(src, sdev);
  if (ddev != host)
    omp_target_free(dst, ddev);
  free(hsrc);
  free(href);
  free(hout);
  return rc;
}

int main() {
  // 1D
  size_t v1[1] = {10}, o1[1] = {3}, s1[1] = {5}, d1[1] = {32};
  // 2D: 3 rows of 5 elements
  size_t v2[2] = {3, 5}, so2[2] = {1, 2}, do2[2] = {2, 1};
  size_t sd2[2] = {6, 9}, dd2[2] = {7, 10};
  // 3D
  size_t v3[3] = {2, 3, 4}, so3[3] = {1, 2, 3}, do3[3] = {2, 1, 0};
  size_t dim3[3] = {4, 6, 8};
  // 4D (handled by the generic per-row path)
  size_t v4[4] = {2, 2, 3, 4}, so4[4] = {1, 0, 1, 2}, do4[4] = {0, 1, 1, 1};
  size_t dim4[4] = {3, 3, 4, 7};

  static const char *dirs[] = {"H2H", "H2D", "D2H", "D2D"};
  char name[64];
  for (int d = H2H; d <= D2D; ++d) {
    for (size_t es = 4; es <= 8; es += 4) {
#define CASE(N, V, DO, SO, DD, SD)                                             \
  snprintf(name, sizeof(name), "%dD %s es=%zu", N, dirs[d], es);               \
  if (run_case(name, d, N, es, V, DO, SO, DD, SD))                             \
    return 1;
      CASE(1, v1, o1, s1, d1, d1)
      CASE(2, v2, do2, so2, dd2, sd2)
      CASE(3, v3, do3, so3, dim3, dim3)
      CASE(4, v4, do4, so4, dim4, dim4)
#undef CASE
    }
  }

  // Nothing to copy: succeeds without looking at the devices.
  size_t vz[2] = {0, 4}, oz[2] = {0, 0}, dz[2] = {4, 4};
  char a, b;
  if (omp_target_memcpy_rect(&a, &b, 1, 2, vz, oz, oz, dz, dz, 12345, 12345)) {
    printf("zero volume failed\n");
    return 1;
  }

  printf("PASS\n");
  return 0;
}

// CHECK: PASS
