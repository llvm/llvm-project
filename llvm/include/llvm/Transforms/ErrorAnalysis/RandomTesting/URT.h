//
// Created by tanmay on 8/29/22.
//

#include "../AtomicCondition/AtomicCondition.h"
#include "../AtomicCondition/AmplificationFactor.h"
#include "../AtomicCondition/ComputationGraph.h"
#include "../ResultStorage/ResultStorage.h"

/*----------------------------------------------------------------------------*/
/* Globals                                                                    */
/*----------------------------------------------------------------------------*/



void fURTf(float steps, float func (float op1), float op1_lower, float op1_upper) {
  for(float i=op1_lower; i<op1_upper; i+=steps) {
//    printf("Result: %f\n", func(i));
    func(i);
  }

  fRSStoreACResult();
}

void fURTff(float steps, float func (float op1, float op2), float op1_lower,
            float op1_upper, float op2_lower, float op2_upper) {
  for(float i=op1_lower; i<op1_upper; i+=steps) {
    for(float j=op2_lower; j<op2_upper; j+=steps) {
//      printf("Result: %f\n", func(i, j));
      func(i, j);
    }
  }

  fRSStoreACResult();
}