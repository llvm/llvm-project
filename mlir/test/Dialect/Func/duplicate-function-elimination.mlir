// RUN: mlir-opt %s --split-input-file --duplicate-function-elimination | \
// RUN: FileCheck %s

func.func @identity(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

func.func @also_identity(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

func.func @yet_another_identity(%arg0: tensor<f32>) -> tensor<f32> {
  return %arg0 : tensor<f32>
}

func.func @user(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = call @identity(%arg0) : (tensor<f32>) -> tensor<f32>
  %1 = call @also_identity(%0) : (tensor<f32>) -> tensor<f32>
  %2 = call @yet_another_identity(%1) : (tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

// CHECK:     @identity
// CHECK-NOT: @also_identity
// CHECK-NOT: @yet_another_identity
// CHECK:     @user
// CHECK-3:     call @identity

// -----

func.func @add_lr(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.addf %arg0, %arg1 : f32
  return %0 : f32
}

func.func @also_add_lr(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.addf %arg0, %arg1 : f32
  return %0 : f32
}

func.func @add_rl(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.addf %arg1, %arg0 : f32
  return %0 : f32
}

func.func @also_add_rl(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.addf %arg1, %arg0 : f32
  return %0 : f32
}

func.func @user(%arg0: f32, %arg1: f32) -> f32 {
  %0 = call @add_lr(%arg0, %arg1) : (f32, f32) -> f32
  %1 = call @also_add_lr(%arg0, %arg1) : (f32, f32) -> f32
  %2 = call @add_rl(%0, %1) : (f32, f32) -> f32
  %3 = call @also_add_rl(%arg0, %2) : (f32, f32) -> f32
 return %3 : f32
}

// CHECK:     @add_lr
// CHECK-NOT: @also_add_lr
// CHECK:     @add_rl
// CHECK-NOT: @also_add_rl
// CHECK:     @user
// CHECK-2:     call @add_lr
// CHECK-2:     call @add_rl

// -----

func.func @ite(%pred: i1, %then: f32, %else: f32) -> f32 {
  %0 = scf.if %pred -> f32 {
    scf.yield %then : f32
  } else {
    scf.yield %else : f32
  }
  return %0 : f32
}

func.func @also_ite(%pred: i1, %then: f32, %else: f32) -> f32 {
  %0 = scf.if %pred -> f32 {
    scf.yield %then : f32
  } else {
    scf.yield %else : f32
  }
  return %0 : f32
}

func.func @reverse_ite(%pred: i1, %then: f32, %else: f32) -> f32 {
  %0 = scf.if %pred -> f32 {
    scf.yield %else : f32
  } else {
    scf.yield %then : f32
  }
  return %0 : f32
}

func.func @user(%pred : i1, %arg0: f32, %arg1: f32) -> f32 {
  %0 = call @also_ite(%pred, %arg0, %arg1) : (i1, f32, f32) -> f32
  %1 = call @ite(%pred, %arg0, %arg1) : (i1, f32, f32) -> f32
  %2 = call @reverse_ite(%pred, %0, %1) : (i1, f32, f32) -> f32
 return %2 : f32
}

// CHECK:     @ite
// CHECK-NOT: @also_ite
// CHECK:     @reverse_ite
// CHECK:     @user
// CHECK-2:     call @ite
// CHECK:       call @reverse_ite

// -----

func.func @deep_tree(%p0: i1, %p1: i1, %p2: i1, %p3: i1, %even: f32, %odd: f32)
    -> f32 {
  %0 = scf.if %p0 -> f32 {
    %1 = scf.if %p1 -> f32 {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    } else {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    }
    scf.yield %1 : f32
  } else {
    %1 = scf.if %p1 -> f32 {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    } else {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    }
    scf.yield %1 : f32
  }
  return %0 : f32
}

func.func @also_deep_tree(%p0: i1, %p1: i1, %p2: i1, %p3: i1, %even: f32,
    %odd: f32) -> f32 {
  %0 = scf.if %p0 -> f32 {
    %1 = scf.if %p1 -> f32 {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    } else {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    }
    scf.yield %1 : f32
  } else {
    %1 = scf.if %p1 -> f32 {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    } else {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    }
    scf.yield %1 : f32
  }
  return %0 : f32
}

func.func @reverse_deep_tree(%p0: i1, %p1: i1, %p2: i1, %p3: i1, %even: f32,
    %odd: f32) -> f32 {
  %0 = scf.if %p0 -> f32 {
    %1 = scf.if %p1 -> f32 {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    } else {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    }
    scf.yield %1 : f32
  } else {
    %1 = scf.if %p1 -> f32 {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    } else {
      %2 = scf.if %p2 -> f32 {
        %3 = scf.if %p3 -> f32 {
          scf.yield %odd : f32
        } else {
          scf.yield %even : f32
        }
        scf.yield %3 : f32
      } else {
        %3 = scf.if %p3 -> f32 {
          scf.yield %even : f32
        } else {
          scf.yield %odd : f32
        }
        scf.yield %3 : f32
      }
      scf.yield %2 : f32
    }
    scf.yield %1 : f32
  }
  return %0 : f32
}

func.func @user(%p0: i1, %p1: i1, %p2: i1, %p3: i1, %odd: f32, %even: f32)
    -> (f32, f32, f32) {
  %0 = call @deep_tree(%p0, %p1, %p2, %p3, %odd, %even)
      : (i1, i1, i1, i1, f32, f32) -> f32
  %1 = call @also_deep_tree(%p0, %p1, %p2, %p3, %odd, %even)
      : (i1, i1, i1, i1, f32, f32) -> f32
  %2 = call @reverse_deep_tree(%p0, %p1, %p2, %p3, %odd, %even)
      : (i1, i1, i1, i1, f32, f32) -> f32
  return %0, %1, %2 : f32, f32, f32
}

// CHECK:     @deep_tree
// CHECK-NOT: @also_deep_tree
// CHECK:     @reverse_deep_tree
// CHECK:     @user
// CHECK-2:     call @deep_tree
// CHECK:       call @reverse_deep_tree
