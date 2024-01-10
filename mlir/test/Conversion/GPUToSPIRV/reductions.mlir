// RUN: mlir-opt -split-input-file -convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupFAdd <Workgroup> <Reduce> %[[ARG]] : f32
    %reduced = gpu.all_reduce add %arg uniform {} : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformFAdd "Workgroup" "Reduce" %[[ARG]] : f32
    %reduced = gpu.all_reduce add %arg {} : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupIAdd <Workgroup> <Reduce> %[[ARG]] : i32
    %reduced = gpu.all_reduce add %arg uniform {} : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformIAdd "Workgroup" "Reduce" %[[ARG]] : i32
    %reduced = gpu.all_reduce add %arg {} : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupFAdd <Subgroup> <Reduce> %[[ARG]] : f32
    %reduced = gpu.subgroup_reduce add %arg uniform : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformFAdd "Subgroup" "Reduce" %[[ARG]] : f32
    %reduced = gpu.subgroup_reduce add %arg : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupIAdd <Subgroup> <Reduce> %[[ARG]] : i32
    %reduced = gpu.subgroup_reduce add %arg uniform : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformIAdd "Subgroup" "Reduce" %[[ARG]] : i32
    %reduced = gpu.subgroup_reduce add %arg : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.KHR.GroupFMul <Workgroup> <Reduce> %[[ARG]] : f32
    %reduced = gpu.all_reduce mul %arg uniform {} : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformFMul "Workgroup" "Reduce" %[[ARG]] : f32
    %reduced = gpu.all_reduce mul %arg {} : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.KHR.GroupIMul <Workgroup> <Reduce> %[[ARG]] : i32
    %reduced = gpu.all_reduce mul %arg uniform {} : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformIMul "Workgroup" "Reduce" %[[ARG]] : i32
    %reduced = gpu.all_reduce mul %arg {} : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.KHR.GroupFMul <Subgroup> <Reduce> %[[ARG]] : f32
    %reduced = gpu.subgroup_reduce mul %arg uniform : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformFMul "Subgroup" "Reduce" %[[ARG]] : f32
    %reduced = gpu.subgroup_reduce mul %arg : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.KHR.GroupIMul <Subgroup> <Reduce> %[[ARG]] : i32
    %reduced = gpu.subgroup_reduce mul %arg uniform : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformIMul "Subgroup" "Reduce" %[[ARG]] : i32
    %reduced = gpu.subgroup_reduce mul %arg : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupFMin <Workgroup> <Reduce> %[[ARG]] : f32
    %reduced = gpu.all_reduce minnumf %arg uniform {} : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformFMin "Workgroup" "Reduce" %[[ARG]] : f32
    %reduced = gpu.all_reduce minnumf %arg {} : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupSMin <Workgroup> <Reduce> %[[ARG]] : i32
    // CHECK: %{{.*}} = spirv.GroupUMin <Workgroup> <Reduce> %[[ARG]] : i32
    %r0 = gpu.all_reduce minsi %arg uniform {} : (i32) -> (i32)
    %r1 = gpu.all_reduce minui %arg uniform {} : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformUMin "Workgroup" "Reduce" %[[ARG]] : i32
    %r0 = gpu.all_reduce minsi %arg {} : (i32) -> (i32)
    %r1 = gpu.all_reduce minui %arg {} : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupFMin <Subgroup> <Reduce> %[[ARG]] : f32
    %reduced = gpu.subgroup_reduce minnumf %arg uniform : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformFMin "Subgroup" "Reduce" %[[ARG]] : f32
    %reduced = gpu.subgroup_reduce minnumf %arg : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupSMin <Subgroup> <Reduce> %[[ARG]] : i32
    // CHECK: %{{.*}} = spirv.GroupUMin <Subgroup> <Reduce> %[[ARG]] : i32
    %r0 = gpu.subgroup_reduce minsi %arg uniform : (i32) -> (i32)
    %r1 = gpu.subgroup_reduce minui %arg uniform : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformSMin "Subgroup" "Reduce" %[[ARG]] : i32
    // CHECK: %{{.*}} = spirv.GroupNonUniformUMin "Subgroup" "Reduce" %[[ARG]] : i32
    %r0 = gpu.subgroup_reduce minsi %arg : (i32) -> (i32)
    %r1 = gpu.subgroup_reduce minui %arg : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupFMax <Workgroup> <Reduce> %[[ARG]] : f32
    %reduced = gpu.all_reduce maxnumf %arg uniform {} : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformFMax "Workgroup" "Reduce" %[[ARG]] : f32
    %reduced = gpu.all_reduce maxnumf %arg {} : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupSMax <Workgroup> <Reduce> %[[ARG]] : i32
    // CHECK: %{{.*}} = spirv.GroupUMax <Workgroup> <Reduce> %[[ARG]] : i32
    %r0 = gpu.all_reduce maxsi %arg uniform {} : (i32) -> (i32)
    %r1 = gpu.all_reduce maxui %arg uniform {} : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformSMax "Workgroup" "Reduce" %[[ARG]] : i32
    // CHECK: %{{.*}} = spirv.GroupNonUniformUMax "Workgroup" "Reduce" %[[ARG]] : i32
    %r0 = gpu.all_reduce maxsi %arg {} : (i32) -> (i32)
    %r1 = gpu.all_reduce maxui %arg {} : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupFMax <Subgroup> <Reduce> %[[ARG]] : f32
    %reduced = gpu.subgroup_reduce maxnumf %arg uniform : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: f32)
  gpu.func @test(%arg : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformFMax "Subgroup" "Reduce" %[[ARG]] : f32
    %reduced = gpu.subgroup_reduce maxnumf %arg : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupSMax <Subgroup> <Reduce> %[[ARG]] : i32
    // CHECK: %{{.*}} = spirv.GroupUMax <Subgroup> <Reduce> %[[ARG]] : i32
    %r0 = gpu.subgroup_reduce maxsi %arg uniform : (i32) -> (i32)
    %r1 = gpu.subgroup_reduce maxui %arg uniform : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : i32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformSMax "Subgroup" "Reduce" %[[ARG]] : i32
    // CHECK: %{{.*}} = spirv.GroupNonUniformUMax "Subgroup" "Reduce" %[[ARG]] : i32
    %r0 = gpu.subgroup_reduce maxsi %arg : (i32) -> (i32)
    %r1 = gpu.subgroup_reduce maxui %arg : (i32) -> (i32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL:  spirv.func @test
  //  CHECK-SAME: (%[[ARG:.*]]: i32)
  gpu.func @test(%arg : vector<1xi32>) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // CHECK: %{{.*}} = spirv.GroupNonUniformSMax "Subgroup" "Reduce" %[[ARG]] : i32
    %r0 = gpu.subgroup_reduce maxsi %arg : (vector<1xi32>) -> (vector<1xi32>)
    gpu.return
  }
}

}

// -----

// TODO: Handle boolean reductions.

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  gpu.func @add(%arg : i1) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // expected-error @+1 {{failed to legalize operation 'gpu.subgroup_reduce'}}
    %r0 = gpu.subgroup_reduce add %arg : (i1) -> (i1)
    gpu.return
  }
}
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {
gpu.module @kernels {
  gpu.func @mul(%arg : i1) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // expected-error @+1 {{failed to legalize operation 'gpu.subgroup_reduce'}}
    %r0 = gpu.subgroup_reduce mul %arg : (i1) -> (i1)
    gpu.return
  }
}
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {
gpu.module @kernels {
  gpu.func @minsi(%arg : i1) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // expected-error @+1 {{failed to legalize operation 'gpu.subgroup_reduce'}}
    %r0 = gpu.subgroup_reduce minsi %arg : (i1) -> (i1)
    gpu.return
  }
}
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {
gpu.module @kernels {
  gpu.func @minui(%arg : i1) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // expected-error @+1 {{failed to legalize operation 'gpu.subgroup_reduce'}}
    %r0 = gpu.subgroup_reduce minui %arg : (i1) -> (i1)
    gpu.return
  }
}
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {
gpu.module @kernels {
  gpu.func @maxsi(%arg : i1) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // expected-error @+1 {{failed to legalize operation 'gpu.subgroup_reduce'}}
    %r0 = gpu.subgroup_reduce maxsi %arg : (i1) -> (i1)
    gpu.return
  }
}
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {
gpu.module @kernels {
  gpu.func @maxui(%arg : i1) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // expected-error @+1 {{failed to legalize operation 'gpu.subgroup_reduce'}}
    %r0 = gpu.subgroup_reduce maxui %arg : (i1) -> (i1)
    gpu.return
  }
}
}

// -----

// Vector reductions need to be lowered to scalar reductions first.

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {
gpu.module @kernels {
  gpu.func @maxui(%arg : vector<2xi32>) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    // expected-error @+1 {{failed to legalize operation 'gpu.subgroup_reduce'}}
    %r0 = gpu.subgroup_reduce maxui %arg : (vector<2xi32>) -> (vector<2xi32>)
    gpu.return
  }
}
}
