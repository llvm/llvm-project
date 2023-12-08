// Check that omp atomic is permitted and behaves when strictly nested within
// omp teams.  This is an extension to OpenMP 5.2 and is enabled by default.

// RUN: %libomp-compile-and-run | FileCheck %s
// GCC has really limited OpenMP 5.2 support yet.
// UNSUPPORTED: gcc

#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

// High parallelism increases our chances of detecting a lack of atomicity.
#define NUM_TEAMS_TRY 256

int main() {
  //      CHECK: update: num_teams=[[#NUM_TEAMS:]]{{$}}
  // CHECK-NEXT: update: x=[[#NUM_TEAMS]]{{$}}
  int x = 0;
  int numTeams;
  #pragma omp teams num_teams(NUM_TEAMS_TRY)
  {
    #pragma omp atomic update
    ++x;
    if (omp_get_team_num() == 0)
      numTeams = omp_get_num_teams();
  }
  printf("update: num_teams=%d\n", numTeams);
  printf("update: x=%d\n", x);

  // CHECK-NEXT: capture: x=[[#NUM_TEAMS]]{{$}}
  // CHECK-NEXT: capture: xCapturedCount=[[#NUM_TEAMS]]{{$}}
  bool xCaptured[numTeams];
  memset(xCaptured, 0, sizeof xCaptured);
  x = 0;
  #pragma omp teams num_teams(NUM_TEAMS_TRY)
  {
    int v;
    #pragma omp atomic capture
    v = x++;
    xCaptured[v] = true;
  }
  printf("capture: x=%d\n", x);
  int xCapturedCount = 0;
  for (int i = 0; i < numTeams; ++i) {
    if (xCaptured[i])
      ++xCapturedCount;
  }
  printf("capture: xCapturedCount=%d\n", xCapturedCount);
  return 0;
}
