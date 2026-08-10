//===- MatmulOptimizer.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_MATMULOPTIMIZER_H
#define POLLY_MATMULOPTIMIZER_H

#include "isl/isl-noexceptions.h"

namespace llvm {
class TargetTransformInfo;
}

namespace polly {
class Dependences;

/// Apply the BLIS matmul optimization pattern if possible.
///
/// Make the loops containing the matrix multiplication be the innermost
/// loops and apply the BLIS matmul optimization pattern. BLIS implements
/// gemm as three nested loops around a macro-kernel, plus two packing
/// routines. The macro-kernel is implemented in terms of two additional
/// loops around a micro-kernel. The micro-kernel is a loop around a rank-1
/// (i.e., outer product) update.
///
/// For a detailed description please see [1].
///
/// The order of the loops defines the data reused in the BLIS implementation
/// of gemm ([1]). In particular, elements of the matrix B, the second
/// operand of matrix multiplication, are reused between iterations of the
/// innermost loop. To keep the reused data in cache, only elements of matrix
/// A, the first operand of matrix multiplication, should be evicted during
/// an iteration of the innermost loop. To provide such a cache replacement
/// policy, elements of the matrix A can, in particular, be loaded first and,
/// consequently, be least-recently-used.
///
/// In our case matrices are stored in row-major order instead of
/// column-major order used in the BLIS implementation ([1]). It affects only
/// on the form of the BLIS micro kernel and the computation of its
/// parameters. In particular, reused elements of the matrix B are
/// successively multiplied by specific elements of the matrix A.
///
/// Refs.:
/// [1] - Analytical Modeling is Enough for High Performance BLIS
/// Tze Meng Low, Francisco D Igual, Tyler M Smith, Enrique S Quintana-Orti
/// Technical Report, 2014
/// http://www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf
///
/// @see ScheduleTreeOptimizer::createMicroKernel
/// @see ScheduleTreeOptimizer::createMacroKernel
/// @see getMicroKernelParams
/// @see getMacroKernelParams
///
/// TODO: Implement the packing transformation.
///
/// @param Node The node that contains a band to be optimized. The node
///             is required to successfully pass
///             ScheduleTreeOptimizer::isMatrMultPattern.
/// @param TTI  Target Transform Info.
/// @param D    The dependencies.
///
/// @returns    The transformed schedule or nullptr if the optimization
///             cannot be applied.
isl::schedule_node
tryOptimizeMatMulPattern(isl::schedule_node Node,
                         const llvm::TargetTransformInfo *TTI,
                         const Dependences *D);

} // namespace polly
#endif // POLLY_MATMULOPTIMIZER_H
