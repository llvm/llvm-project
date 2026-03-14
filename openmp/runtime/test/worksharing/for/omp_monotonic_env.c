// RUN: %libomp-compile
// RUN: env OMP_SCHEDULE=monotonic:dynamic,50 %libomp-run monotonic dynamic 50
// RUN: env OMP_SCHEDULE=monotonic:guided,51 %libomp-run monotonic guided 51
// RUN: env OMP_SCHEDULE=monotonic:static,52 %libomp-run monotonic static 52
// RUN: env OMP_SCHEDULE=nonmonotonic:dynamic,53 %libomp-run nonmonotonic dynamic 53
// RUN: env OMP_SCHEDULE=nonmonotonic:guided,54 %libomp-run nonmonotonic guided 54

// The test checks OMP 5.0 monotonic/nonmonotonic OMP_SCHEDULE parsing
// The nonmonotonic tests see if the parser accepts nonmonotonic, if the
// parser doesn't then a static schedule is assumed

#include <stdio.h>
#include <string.h>
#include <omp.h>

int err = 0;

omp_sched_t sched_without_modifiers(omp_sched_t sched) {
  return (omp_sched_t)((int)sched & ~((int)omp_sched_monotonic));
}

int sched_has_modifiers(omp_sched_t sched, omp_sched_t modifiers) {
  return (int)sched & (int)modifiers;
}

// check that sched = hope | modifiers
void check_schedule(const char *extra, const omp_sched_t sched, int chunk,
                    omp_sched_t hope_sched, int hope_chunk) {

  if (sched != hope_sched || chunk != hope_chunk) {
    ++err;
    printf("Error: %s: schedule: (%d, %d) is not equal to (%d, %d)\n", extra,
           (int)hope_sched, hope_chunk, (int)sched, chunk);
  }
}

omp_sched_t str2omp_sched(const char *str) {
  if (!strcmp(str, "dynamic"))
    return omp_sched_dynamic;
  if (!strcmp(str, "static"))
    return omp_sched_static;
  if (!strcmp(str, "guided"))
    return omp_sched_guided;
  printf("Error: Unknown schedule type: %s\n", str);
  exit(1);
}

int is_monotonic(const char *str) { return !strcmp(str, "monotonic"); }

int main(int argc, char **argv) {
  int i, monotonic, chunk, ref_chunk;
  omp_sched_t sched, ref_sched;

  if (argc != 4) {
    printf("Error: usage: <executable> monotonic|nonmonotonic <schedule> "
           "<chunk-size>\n");
    exit(1);
  }

  monotonic = is_monotonic(argv[1]);
  ref_sched = str2omp_sched(argv[2]);
  ref_chunk = atoi(argv[3]);

  omp_get_schedule(&sched, &chunk);

  if (monotonic && !sched_has_modifiers(sched, omp_sched_monotonic)) {
    printf("Error: sched (0x%x) does not have monotonic modifier\n",
           (int)sched);
    ++err;
  }
  sched = sched_without_modifiers(sched);
  if (sched != ref_sched) {
    printf("Error: sched (0x%x) is not 0x%x\n", (int)sched, (int)ref_sched);
    ++err;
  }
  if (chunk != ref_chunk) {
    printf("Error: chunk is not %d\n", ref_chunk);
    ++err;
  }
  if (err > 0) {
    printf("Failed\n");
    return 1;
  }
  printf("Passed\n");
  return 0;
}
