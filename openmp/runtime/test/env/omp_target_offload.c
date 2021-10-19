// RUN: %libomp-compile-and-run
#include <string.h>
#include <stdlib.h>

enum kmp_target_offload_kind {
  tgt_disabled = 0,
  tgt_default = 1,
  tgt_mandatory = 2
};

extern int __kmpc_get_target_offload();
extern void kmp_set_defaults(char const *str);

const char *disabled_examples[] = {
    // Allowed inputs
    "disabled", "DISABLED", "Disabled", "dIsAbLeD", "DiSaBlEd"};

const char *default_examples[] = {
    // Allowed inputs
    "default", "DEFAULT", "Default", "deFAulT", "DEfaULt",
    // These should be changed to default (failed match)
    "mandatry", "defaults", "disable", "enabled", "mandatorynot"};

const char *mandatory_examples[] = {
    // Allowed inputs
    "mandatory", "MANDATORY", "Mandatory", "manDatoRy", "MANdATOry"};

// Return target-offload-var ICV
int get_target_offload_icv() {
#pragma omp parallel
  {}
  return __kmpc_get_target_offload();
}

int main() {
  int i;
  const char *omp_target_offload = "OMP_TARGET_OFFLOAD=";
  char buf[80];

  for (i = 0; i < sizeof(disabled_examples) / sizeof(char *); ++i) {
    strcpy(buf, omp_target_offload);
    strcat(buf, disabled_examples[i]);
    kmp_set_defaults(buf);
    if (tgt_disabled != get_target_offload_icv())
      return EXIT_FAILURE;
  }
  for (i = 0; i < sizeof(default_examples) / sizeof(char *); ++i) {
    strcpy(buf, omp_target_offload);
    strcat(buf, default_examples[i]);
    kmp_set_defaults(buf);
    if (tgt_default != get_target_offload_icv())
      return EXIT_FAILURE;
  }
  for (i = 0; i < sizeof(mandatory_examples) / sizeof(char *); ++i) {
    strcpy(buf, omp_target_offload);
    strcat(buf, mandatory_examples[i]);
    kmp_set_defaults(buf);
    if (tgt_mandatory != get_target_offload_icv())
      return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
