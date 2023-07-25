//===- Barvinok.cpp - Barvinok's Algorithm -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Barvinok.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include <numeric>
#include <optional>

using namespace mlir;
using namespace presburger;

SmallVector<std::pair<int, ConeH>, 16> unimodularDecomposition(ConeH cone)
{
    ConeV dualCone = getDual(cone);
    SmallVector<std::pair<int, ConeV>, 16> dualDecomposed;
    SmallVector<std::pair<int, ConeH>, 16> decomposed;

    MPInt index = getIndex(dualCone);
    if (index == 0)
    {
        SmallVector<ConeV, 16> simplicialCones = triangulate(dualCone);
        for (ConeV simplicialCone : simplicialCones)
        {
            SmallVector<std::pair<int, ConeV>, 16> unimodularCones = unimodularDecompositionSimplicial(1, simplicialCone);
            dualDecomposed.append(unimodularCones);
        }
    }
    else
    {
        dualDecomposed = unimodularDecompositionSimplicial(1, dualCone);
    }
    
    for (std::pair<int, ConeV> dualComponent : dualDecomposed)
    {
        decomposed.append(1, std::make_pair(dualComponent.first, getDual(dualComponent.second)));
    }

    return decomposed;
    
}