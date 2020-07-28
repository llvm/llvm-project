//####This pass does the loop interchange#####//
// This code makes the assumption that if the input program have a loop nest
// then that loop nest will contain atleast one array load or store operation

#include "PassDetail.h"
#include "algorithm"
#include "bits/stdc++.h"
#include "iostream"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "typeinfo"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

#define DEBUG_TYPE "affine-loop-interchange"

namespace {

class ValuePositionMap {
public:
  void addSrcValue(Value value) {
    if (addValueAt(value, &srcDimPosMap, numSrcDims))
      ++numSrcDims;
  }
  void addSymbolValue(Value value) {
    if (addValueAt(value, &symbolPosMap, numSymbols))
      ++numSymbols;
  }
  unsigned getSrcDimOrSymPos(Value value) const {
    return getDimOrSymPos(value, srcDimPosMap, 0);
  }
  unsigned getSymPos(Value value) const {
    auto it = symbolPosMap.find(value);
    assert(it != symbolPosMap.end());
    return numSrcDims + it->second;
  }

  unsigned getNumSrcDims() const { return numSrcDims; }
  unsigned getNumDims() const { return numSrcDims; }
  unsigned getNumSymbols() const { return numSymbols; }

private:
  bool addValueAt(Value value, DenseMap<Value, unsigned> *posMap,
                  unsigned position) {
    auto it = posMap->find(value);
    if (it == posMap->end()) {
      (*posMap)[value] = position;
      return true;
    }
    return false;
  }
  unsigned getDimOrSymPos(Value value,
                          const DenseMap<Value, unsigned> &dimPosMap,
                          unsigned dimPosOffset) const {
    auto it = dimPosMap.find(value);
    if (it != dimPosMap.end()) {
      return dimPosOffset + it->second;
    }
    it = symbolPosMap.find(value);
    assert(it != symbolPosMap.end());
    return numSrcDims + it->second;
  }

  unsigned numSrcDims = 0;
  unsigned numSymbols = 0;
  DenseMap<Value, unsigned> srcDimPosMap;
  DenseMap<Value, unsigned> symbolPosMap;
};

// Loop Interchange Pass
struct LoopInterchange : public AffineLoopInterchangeBase<LoopInterchange> {
  SmallVector<Operation *, 4> loadsAndStores;
  // A vector for storing operation type i.e. load or store
  std::vector<char> oper_type;
  // A 3d vector used for storing access matrix
  std::vector<std::vector<std::vector<int64_t>>> accessmat;
  // A 2d vector for storing dependences
  std::vector<std::vector<int64_t>> dependencemat;
  // This vector contains the rank of all access matrices of Operations
  // in the order of their apperance in the program
  std::vector<int> rank_accmat;
  // This variable contains the rank of dependence Matrix
  int rank_depmat;
  // Variables for access AccessMatrix
  unsigned page, rows, col;
  // This vector contains all valid loop permutations
  std::vector<std::vector<int>> validperms;
  // Variable to tell total no. of valid permutations
  int num_vp;
  // this vector contains the indexes of loops which can be parallelized
  std::vector<int> par_loops;
  // Vector to contain groups formed
  std::vector<std::set<int>> final_groups;
  // contains index of operations with spatial locality
  std::set<int> sl_ind;
  // contains index of operations with temporal locality
  std::set<int> tl_ind;
  // variable for maintaing dependencemat index
  int dm_index;

  void runOnFunction() override;
  void find_dep_access(AffineForOp floop);
  void createAccessMatrix(ArrayRef<Operation *> loadsAndStores);
  void checkDependences(ArrayRef<Operation *> loadsAndStores);
  std::string
  getDirectionVectorStr(bool ret, unsigned numCommonLoops,
                        unsigned loopNestDepth,
                        ArrayRef<DependenceComponent> dependenceComponents);
  void valid_perm();
  void checkvalidperm(int ar[], int loop_dep);
  int best_perm();
  std::vector<int> cost_model(std::vector<int> cperm);
  int spatial_score(std::vector<int> perm_ss);
  int temporal_score(std::vector<int> perm_ts);
  int parallelism_score(std::vector<int> perm_ps);
  void find_parallel_loops();
  int group_reuse();
  bool compare_accmat(int i, int j);
  void form_group();
};
} // end of namespace

// checks if t1 is a proper subset(equal as well) of t2 or not
bool isSubset(std::set<int> t1, std::set<int> t2) {
  if (t1.size() >= t2.size())
    return false;
  bool flag = true;
  std::set<int>::iterator it1;
  for (it1 = t1.begin(); it1 != t1.end(); it1++) {
    if (t2.find(*it1) == t2.end()) {
      flag = false;
      break;
    }
  }
  return flag;
}

bool LoopInterchange::compare_accmat(int i, int j) {
  int size_i = (accessmat[i].size()) * (accessmat[i][0].size());
  int size_j = (accessmat[j].size()) * (accessmat[j][0].size());
  if (size_i != size_j)
    return false;
  bool flag = true;
  for (int p = 0, e = accessmat[i].size(); p < e; p++) {
    for (int q = 0, k = accessmat[i][p].size(); q < k; q++) {
      if (accessmat[i][p][q] != accessmat[j][p][q]) {
        flag = false;
        break;
      }
    }
    if (!flag)
      break;
  }
  return flag;
}

// This function forms the groups for a loop nest in the input program
void LoopInterchange::form_group() {
  final_groups.clear();
  std::vector<std::set<int>> groups;
  for (int i = 0, e = accessmat.size(); i < e; i++) {
    for (int j = 0, f = e; j < f; j++) {
      bool eq_accmat = compare_accmat(i, j);
      auto *srcOpInst = loadsAndStores[i];
      MemRefAccess srcAccess(srcOpInst);
      auto *dstOpInst = loadsAndStores[j];
      MemRefAccess dstAccess(dstOpInst);
      unsigned numCommonLoops =
          getNumCommonSurroundingLoops(*srcOpInst, *dstOpInst);
      bool dep_flag = false;
      for (unsigned d = 1; d <= numCommonLoops + 1; ++d) {
        FlatAffineConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> dependenceComponents;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, &dependenceConstraints,
            &dependenceComponents, true);
        if (hasDependence(result)) {
          dep_flag = true;
          break;
        }
      }
      if (eq_accmat && dep_flag) { // true if ith access matrix is equal to jth
        // if there's a dependence b/w i & j
        bool in_flag = false;
        for (int p = 0, g = groups.size(); p < g; p++) {
          if (groups[p].find(i) != groups[p].end() ||
              groups[p].find(j) != groups[p].end()) {
            groups[p].insert(i);
            groups[p].insert(j);
            in_flag = true;
            break;
          }
        }
        if (!in_flag) {
          std::set<int> t;
          t.insert(i);
          t.insert(j);
          groups.push_back(t);
        }
      } else {
        bool out_f1 = false;
        bool out_f2 = false;
        for (int p = 0, g = groups.size(); p < g; p++) {
          if (groups[p].find(i) != groups[p].end()) {
            out_f1 = true;
          }
          if (groups[p].find(j) != groups[p].end()) {
            out_f2 = true;
          }
        }
        if (!out_f1) {
          std::set<int> t1;
          t1.insert(i);
          groups.push_back(t1);
        }

        if (!out_f2) {
          std::set<int> t2;
          t2.insert(j);
          groups.push_back(t2);
        }
      }
    }
  }
  // eliminating redundancies
  for (int i = 0, e = groups.size(); i < e; i++) {
    bool sub = false;
    for (int j = 0, f = groups.size(); j < f; j++) {
      if (isSubset(groups[i], groups[j])) {
        sub = true;
        break;
      }
    }
    if (!sub)
      final_groups.push_back(groups[i]);
  }
}

// This function finds out the loops that can be parallelized and stores them
// vector par_loops
void LoopInterchange::find_parallel_loops() {
  par_loops.clear();
  std::set<int> seq_loops;
  int n_loops = (int)accessmat[0][0].size();
  for (int i = 0, e = dependencemat.size(); i < e; i++) {
    for (int j = 0, f = dependencemat[i].size(); j < f; j++) {
      if (dependencemat[i][j] > 0) {
        seq_loops.insert(j);
        break;
      }
    }
  }
  for (int i = 0; i < n_loops; i++) {
    if (seq_loops.find(i) == seq_loops.end()) {
      par_loops.push_back(i);
    }
  }
}

// This function computes the group reuse scores
int LoopInterchange::group_reuse() {
  int ts = 0;
  for (int i = 0, e = final_groups.size(); i < e; i++) {
    if (final_groups[i].size() > 1) {
      // enter here only if the group size is greater than one
      // take a representative element from a group let's say 1st element always
      auto it = final_groups[i].begin();
      int rep_ele = *it;
      int temp_sc = 0;
      int spat_sc = 0;
      if (tl_ind.find(rep_ele) != tl_ind.end())
        temp_sc = final_groups[i].size();
      if (sl_ind.find(rep_ele) != sl_ind.end())
        spat_sc = final_groups[i].size();
      ts += temp_sc + spat_sc;
    }
  }
  return ts;
}

// This function computes the sync-free parallelism score for the given
// permutaion
int LoopInterchange::parallelism_score(std::vector<int> perm_ps) {
  // total score for all accesses
  int ts = 0;
  // Check if there are loops that can be parallelized
  if (par_loops.size() > 0) {
    for (int i = 0, e = (int)perm_ps.size(); i < e; i++) {
      if (std::find(par_loops.begin(), par_loops.end(), perm_ps[i]) !=
          par_loops.end()) {
        // true if loop at index i in perm_ps is parallel
        ts = e - i;
      } else {
        if (ts == 0)
          continue;
        break;
      }
    }
  }
  return ts;
}

// This function computes the temporal locality score for the given permutaion
int LoopInterchange::temporal_score(std::vector<int> perm_ts) {
  tl_ind.clear();
  // total score for all accesses
  int ts = 0;
  int cols = perm_ts.size();
  for (int i = 0, e = rank_accmat.size(); i < e; i++) {
    int max_tempLoc = cols - rank_accmat[i];
    for (int j = 1; j <= max_tempLoc; j++) {
      bool flag = true;
      for (int k = 0, f = accessmat[i].size(); k < f; k++) {
        if (accessmat[i][k][perm_ts[cols - j]] != 0) {
          flag = false;
          break;
        }
      }
      if (flag) {
        tl_ind.insert(i);
        ts += 1;
      }
    }
  }
  return ts;
}

// This function computes the spatial locality score for the given permutaion
int LoopInterchange::spatial_score(std::vector<int> perm_ss) {
  sl_ind.clear();
  // total score for all accesses
  int ts = 0;
  // compute the spatial_score for each access matrix
  for (int i = 0, e = accessmat.size(); i < e; i++) {
    // fvd is fastest varying dimension
    int size = perm_ss.size() - 1;
    int fvd = perm_ss[size];
    int index = accessmat[i].size();
    do {
      for (int j = 0, f = accessmat[i].size(); j < f; j++) {
        if (accessmat[i][j][fvd] > 0) {
          index = j;
          break;
        }
      }
      if (index == (int)accessmat[i].size()) { // it means entire column is 0
        size -= 1;
        fvd = perm_ss[size];
        continue;
      } else { // it means a non-zero element in the fvd column
        for (int r = size; r < (int)perm_ss.size(); r++) {
          if (index == ((int)accessmat[i].size()) - 1) {
            sl_ind.insert(i);
            ts += 1;
            break;
          }
        }
        break;
      }
    } while (true);
  }
  return ts;
}

// This function return the cost i.e. score for a valid permutation
std::vector<int> LoopInterchange::cost_model(std::vector<int> cperm) {
  std::vector<int> score;
  score.push_back(spatial_score(cperm));
  score.push_back(temporal_score(cperm));
  score.push_back(parallelism_score(cperm));
  score.push_back(group_reuse());
  return score;
}

// This function finds the best loop permutation out of all valid permutations
// and returns the index of best permutation
int LoopInterchange::best_perm() {
  // This function finds the loops that can be parallelized i.e. loops that
  // don't carry dependences and stores them into vector par_loops
  find_parallel_loops();

  // this function will form the groups and store them in vector final_groups
  form_group();

  // computing the cost for each valid permutation and storing them in the
  // 2d vector cost in the order of spatial,temporal and parallelism score resp.
  std::vector<std::vector<int>> cost;
  for (int i = 0; i < num_vp; i++) {
    std::vector<int> bperm;
    for (int j = 0, c = validperms[i].size(); j < c; j++) {
      bperm.push_back(validperms[i][j]);
    }
    cost.push_back(cost_model(bperm));
  }

  int tempScore, bestScore = -100, bestIndex = 0;
  // finding the index of best loop permutation
  for (int i = 0, e = cost.size(); i < e; i++) {
    tempScore = cost[i][0] + cost[i][1] + cost[i][2];
    if (tempScore > bestScore) {
      bestScore = tempScore;
      bestIndex = i;
    } else if (tempScore == bestScore) {
      if (cost[i][0] + cost[i][1] > cost[bestIndex][0] + cost[bestIndex][1]) {
        // breaking the clash based on locality
        bestIndex = i;
      } else if (cost[i][0] + cost[i][1] ==
                 cost[bestIndex][0] + cost[bestIndex][1]) {
        // breaking the clash based on group reuse
        if (cost[i][3] > cost[bestIndex][3])
          bestIndex = i;
      }
    }
  }
  return bestIndex;
}

// This function checks whether the given permutation is valid or not
void LoopInterchange::checkvalidperm(int ar[], int loop_dep) {
  bool flag = true;
  // check if there's atleast 1 dependence
  if (dependencemat.size() > 0) {
    for (int i = 0, e = dependencemat.size(); i < e; i++) {
      int sum = 0;
      for (int j = 0; j < loop_dep; j++) {
        sum += dependencemat[i][ar[j]];
        if (sum != 0) {
          if (sum < 0) {
            flag = false;
          }
          break;
        }
      }
      if (!flag)
        break;
    }
  }
  if (flag) {
    std::vector<int> vp2;
    validperms.push_back(vp2);
    for (int k = 0; k < loop_dep; k++)
      validperms[num_vp].push_back(ar[k]);
    num_vp++;
  }
}

// This function generates all valid permutations of loop nest
void LoopInterchange::valid_perm() {
  num_vp = 0;
  validperms.clear();
  int num_loop = accessmat[0][0].size();
  int *loopiv{new int[num_loop]{}};
  for (int i = 0; i < num_loop; i++) {
    loopiv[i] = i;
  }
  do {
    checkvalidperm(loopiv, num_loop);
  } while (std::next_permutation(loopiv, loopiv + num_loop));
}

// This function is called by checkDependences function
std::string LoopInterchange::getDirectionVectorStr(
    bool ret, unsigned numCommonLoops, unsigned loopNestDepth,
    ArrayRef<DependenceComponent> dependenceComponents) {
  if (!ret)
    return "false";
  if (dependenceComponents.empty() || loopNestDepth > numCommonLoops)
    return "true";
  std::vector<int64_t> d2;
  dependencemat.push_back(d2);
  std::string result;
  for (unsigned i = 0, e = dependenceComponents.size(); i < e; ++i) {
    int64_t lowerb = 0, temp = 0;
    if (dependenceComponents[i].lb.hasValue() &&
        dependenceComponents[i].lb.getValue() !=
            std::numeric_limits<int64_t>::min())
      lowerb = dependenceComponents[i].lb.getValue();

    int64_t upperb = 0;
    if (dependenceComponents[i].ub.hasValue() &&
        dependenceComponents[i].ub.getValue() !=
            std::numeric_limits<int64_t>::max()) {
      upperb = dependenceComponents[i].ub.getValue();
    }

    if (lowerb == upperb) {
      temp = (lowerb);
    } else if (lowerb < 0 && upperb < 0) {
      temp = (upperb);
    } else if (lowerb > 0 && upperb > 0) {
      temp = (lowerb);
    } else if (lowerb == 0 && upperb > 0) {
      temp = (lowerb + 1);
    } else if (upperb == 0 && lowerb < 0) {
      temp = (upperb - 1);
    } else if (lowerb < 0 && upperb > 0) {
      temp = -1;
    }

    dependencemat[dm_index].push_back(temp);
    result += "[" + std::to_string(temp) + "]";
  }
  return result;
}

// This function check for dependences present inside the loop
void LoopInterchange::checkDependences(ArrayRef<Operation *> loadsAndStores) {
  dm_index = -1;
  for (unsigned i = 0, e = loadsAndStores.size(); i < e; ++i) {
    auto *srcOpInst = loadsAndStores[i];
    MemRefAccess srcAccess(srcOpInst);
    for (unsigned j = 0; j < e; ++j) {
      auto *dstOpInst = loadsAndStores[j];
      MemRefAccess dstAccess(dstOpInst);

      unsigned numCommonLoops =
          getNumCommonSurroundingLoops(*srcOpInst, *dstOpInst);
      for (unsigned d = 1; d <= numCommonLoops; ++d) {
        FlatAffineConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> dependenceComponents;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, &dependenceConstraints,
            &dependenceComponents);

        bool ret = hasDependence(result);
        if (ret) { // This means dependence exists
          dm_index++;
          std::string dep_result = getDirectionVectorStr(ret, numCommonLoops, d,
                                                         dependenceComponents);
          break;
        }
      }
    }
  }
}

// The code from here uptill next 115 lines is taken from the AffineAnalysis.cpp file
static LogicalResult
addMemRefAccessConstraints(const AffineValueMap &srcAccessMap,
                           const ValuePositionMap &valuePosMap,
                           FlatAffineConstraints *dependenceDomain) {
  AffineMap srcMap = srcAccessMap.getAffineMap();
  unsigned numResults = srcMap.getNumResults();

  ArrayRef<Value> srcOperands = srcAccessMap.getOperands();

  std::vector<SmallVector<int64_t, 8>> srcFlatExprs;
  FlatAffineConstraints srcLocalVarCst;
  if (failed(getFlattenedAffineExprs(srcMap, &srcFlatExprs, &srcLocalVarCst)))
    return failure();

  // Equality to add.
  SmallVector<int64_t, 8> eq(dependenceDomain->getNumCols());
  for (unsigned i = 0; i < numResults; ++i) {
    // Zero fill.
    std::fill(eq.begin(), eq.end(), 0);

    // Flattened AffineExpr for src result 'i'.
    const auto &srcFlatExpr = srcFlatExprs[i];
    // Set identifier coefficients from src access function.
    for (unsigned j = 0, e = srcOperands.size(); j < e; ++j)
      eq[valuePosMap.getSrcDimOrSymPos(srcOperands[j])] = srcFlatExpr[j];

    // Add equality constraint.
    dependenceDomain->addEquality(eq);
  }
  return success();
}

static void
buildDimAndSymbolPositionMaps(const FlatAffineConstraints &srcDomain,
                              const AffineValueMap &srcAccessMap,
                              ValuePositionMap *valuePosMap,
                              FlatAffineConstraints *dependenceConstraints) {
  auto updateValuePosMap = [&](ArrayRef<Value> values, bool isSrc) {
    for (unsigned i = 0, e = values.size(); i < e; ++i) {
      auto value = values[i];
      if (isSrc) {
        valuePosMap->addSrcValue(value);
      }
    }
  };

  SmallVector<Value, 4> srcValues, destValues;
  srcDomain.getIdValues(0, srcDomain.getNumDimIds(), &srcValues);
  // Update value position map with identifiers from src iteration domain.
  updateValuePosMap(srcValues, /*isSrc=*/true);
  // Update value position map with identifiers from src access function.
  updateValuePosMap(srcAccessMap.getOperands(), /*isSrc=*/true);
}

// Sets up dependence constraints columns appropriately
static void
initDependenceConstraints(const FlatAffineConstraints &srcDomain,
                          const AffineValueMap &srcAccessMap,
                          const ValuePositionMap &valuePosMap,
                          FlatAffineConstraints *dependenceConstraints) {
  // Calculate number of equalities/inequalities and columns required to
  // initialize FlatAffineConstraints for 'dependenceDomain'.
  unsigned numIneq = srcDomain.getNumInequalities();
  AffineMap srcMap = srcAccessMap.getAffineMap();
  unsigned numEq = srcMap.getNumResults();
  unsigned numDims = srcDomain.getNumDimIds();
  unsigned numSymbols = 0 /*valuePosMap.getNumSymbols()*/;
  unsigned numLocals = 0 /*srcDomain.getNumLocalIds()*/;
  unsigned numIds = numDims + numSymbols + numLocals;
  unsigned numCols = numIds + 1;

  // Set flat affine constraints sizes and reserving space for constraints.
  dependenceConstraints->reset(numIneq, numEq, numCols, numDims, numSymbols,
                               numLocals);

  // Set values corresponding to dependence constraint identifiers.
  SmallVector<Value, 4> srcLoopIVs;
  srcDomain.getIdValues(0, srcDomain.getNumDimIds(), &srcLoopIVs);

  dependenceConstraints->setIdValues(0, srcLoopIVs.size(), srcLoopIVs);
}

static LogicalResult getInstIndexSet(Operation *op,
                                     FlatAffineConstraints *indexSet) {
  SmallVector<AffineForOp, 4> loops;
  getLoopIVs(*op, &loops);
  return getIndexSet(loops, indexSet);
}

// function that computes access matrix for an operation
void checkMemrefAccess(
    const MemRefAccess &srcAccess, unsigned loopDepth,
    FlatAffineConstraints *dependenceConstraints,
    SmallVector<DependenceComponent, 2> *dependenceComponents) {

  // Get composed access function for 'srcAccess'.
  AffineValueMap srcAccessMap;
  srcAccess.getAccessMap(&srcAccessMap);

  // Get iteration domain for the 'srcAccess' operation.
  FlatAffineConstraints srcDomain;
  getInstIndexSet(srcAccess.opInst, &srcDomain);

  ValuePositionMap valuePosMap;
  buildDimAndSymbolPositionMaps(srcDomain, srcAccessMap, &valuePosMap,
                                dependenceConstraints);

  initDependenceConstraints(srcDomain, srcAccessMap, valuePosMap,
                            dependenceConstraints);

  // assert(valuePosMap.getNumDims() == srcDomain.getNumDimIds());

  addMemRefAccessConstraints(srcAccessMap, valuePosMap, dependenceConstraints);
}

// This function creates access matrix for each load and store Operation
void LoopInterchange::createAccessMatrix(ArrayRef<Operation *> loadsAndStores) {
  for (unsigned i = 0, e = loadsAndStores.size(); i < e; ++i) {
    auto *srcOpInst = loadsAndStores[i];
    MemRefAccess srcAccess(srcOpInst);

    FlatAffineConstraints accessConstraints;
    SmallVector<DependenceComponent, 2> accessComponents;
    checkMemrefAccess(srcAccess, 1, &accessConstraints, &accessComponents);
    col = accessConstraints.getNumCols();
    rows = accessConstraints.getNumEqualities();
    col = (col - 1);
    std::vector<std::vector<int64_t>> l2;
    accessmat.push_back(l2);
    for (unsigned r = 0; r < rows; r++) {
      std::vector<int64_t> l3;
      accessmat[i].push_back(l3);
      for (unsigned c = 0; c < col; c++) {
        accessmat[i][r].push_back(accessConstraints.atEq(r, c));
      }
    }
  }
}

// Function to find the rank of access matrix
int compute_rank(std::vector<std::vector<double>> A) {
  int n = A.size();
  int m = A[0].size();
  const double EPS = 1E-9;
  int rank = 0;
  std::vector<bool> row_selected(n, false);
  for (int i = 0; i < m; ++i) {
    int j;
    for (j = 0; j < n; ++j) {
      if (!row_selected[j] && std::abs(A[j][i]) > EPS)
        break;
    }

    if (j != n) {
      ++rank;
      row_selected[j] = true;
      for (int p = i + 1; p < m; ++p)
        A[j][p] /= A[j][i];
      for (int k = 0; k < n; ++k) {
        if (k != j && std::abs(A[k][i]) > EPS) {
          for (int p = i + 1; p < m; ++p)
            A[k][p] -= A[j][p] * A[k][i];
        }
      }
    }
  }
  return rank;
}

// This function will find out all the dependences, access matrices and ranks
// of operations present inside the loop floop
void LoopInterchange::find_dep_access(AffineForOp floop) {
  loadsAndStores.clear();
  oper_type.clear();
  accessmat.clear();
  dependencemat.clear();
  rank_accmat.clear();

  // Finding out all affine loadsAndStores operations
  floop.walk([&](Operation *op) {
    if (isa<AffineLoadOp>(op)) {
      loadsAndStores.push_back(op);
      oper_type.push_back('L');
    } else if (isa<AffineStoreOp>(op)) {
      loadsAndStores.push_back(op);
      oper_type.push_back('S');
    }
  });

  // This function will find out all access matrices and store them in accessmat
  createAccessMatrix(loadsAndStores);

  // Finding out the rank of each access AccessMatrix and storing in rank_accmat
  for (size_t i = 0, e = accessmat.size(); i < e; i++) {
    std::vector<std::vector<double>> am;
    for (size_t j = 0; j < accessmat[i].size(); j++) {
      std::vector<double> am2;
      am.push_back(am2);
      for (size_t k = 0; k < accessmat[i][j].size(); k++) {
        am[j].push_back(accessmat[i][j][k]);
      }
    }
    rank_accmat.push_back(compute_rank(am));
  }

  // This function will find out all the dependences in a loop nest
  // and store them in dependencemat
  checkDependences(loadsAndStores);
  if (dependencemat.size() > 0) {
    // Converting the dependencemat from int to double for rank calculation
    std::vector<std::vector<double>> dm;
    for (size_t i = 0, e = dependencemat.size(); i < e; i++) {
      std::vector<double> dm2;
      dm.push_back(dm2);
      for (size_t j = 0, f = dependencemat[i].size(); j < f; j++) {
        dm[i].push_back(dependencemat[i][j]);
      }
    }
    rank_depmat = compute_rank(dm);
  }
} // end of the find_dep_access function

// This function permutes the given loop nest for the given best permutation
void bestLoopPermute(std::vector<int> int_per, AffineForOp forloop) {
  SmallVector<AffineForOp, 4> allforloops;

  getPerfectlyNestedLoops(allforloops, forloop);

  // conveting permutation from int to unsigned int
  std::vector<unsigned int> permut;

  for (int i = 0, e = int_per.size(); i < e; i++) {
    std::vector<int>::iterator it =
        std::find(int_per.begin(), int_per.end(), i);
    permut.push_back(std::distance(int_per.begin(), it));
  }

  unsigned outerIndex = permuteLoops(allforloops, permut);
  outerIndex += 1;
}

// This function checks whether the given loop nest is rectangular or not
// returns true if loop nest is rectangular
bool checkRect(AffineForOp floop) {
  bool flag = true;
  floop.walk([&](AffineForOp T) {
    if (!(T.hasConstantBounds()))
      flag = false;
  });
  return flag;
}

// This function makes the given imperfect loop nest a perfect loop nest
// works for imperfect loop nests assuming only 2 child loops in parallel
void makeperfect_nest(AffineForOp floop) {
  // creating a for loop using opbuilder same as floop
  OpBuilder opb(floop.getOperation()->getBlock(),
                std::next(Block::iterator(floop.getOperation())));
  AffineForOp clone_op = static_cast<AffineForOp>(opb.clone(*floop));

  // floop_child is the immediate child of floop and same is clone_op_child
  AffineForOp floop_child, clone_op_child;
  int iter = 0;
  floop.walk([&](AffineForOp temp) {
    Operation *par_op = temp.getParentOp();
    if (isa<AffineForOp>(*par_op) &&
        static_cast<AffineForOp>(par_op) == floop) {
      floop_child = temp;
      iter++;
    }
  });

  floop_child.erase();

  int iter2 = 0;
  clone_op.walk([&](AffineForOp temp) {
    Operation *par_op = temp.getParentOp();
    if (isa<AffineForOp>(*par_op) &&
        static_cast<AffineForOp>(par_op) == clone_op) {
      iter2++;
      if (iter2 == iter - 1)
        temp.erase();
    }
  });
}

// This function checks that the given loop nest is perfectly nested or not
bool checkperfectnest(AffineForOp floop) {
  bool flag = false;
  SmallVector<AffineForOp, 4> all_loops;
  Block *loopbody = floop.getBody();
  loopbody->walk([&](AffineForOp temp) {
    Operation *par_temp = temp.getParentOp();
    if (isa<AffineForOp>(*par_temp) &&
        static_cast<AffineForOp>(par_temp) == floop)
      all_loops.push_back(temp);
  });
  all_loops.push_back(floop);
  std::reverse(all_loops.begin(), all_loops.end());
  if (isPerfectlyNested(all_loops))
    flag = true;

  return flag;
}

// This function checks whether the loop nest contains if statement or not
bool checkif(AffineForOp floop) {
  bool flag = false;
  Block *loopbody = floop.getBody();
  loopbody->walk([&](AffineIfOp temp) { flag = true; });
  return flag;
}

// check for all the imperfect loop nests present inside the function fun and
// makes them perfect loop nests
void check_make_perfect_loop(FuncOp fun) {
  bool flag;
  do {
    flag = true;
    fun.walk([&](AffineForOp temp) {
      // true if the loop nest is not perfect
      if (!checkif(temp) && checkRect(temp) && !checkperfectnest(temp)) {
        makeperfect_nest(temp);
        flag = false;
      }
    });
  } while (!flag);
}

void LoopInterchange::runOnFunction() {
  FuncOp func = getFunction();
  // This function makes all the imperfect loop nests perfect loop nests
  // present inside the function func
  check_make_perfect_loop(func);

  // this variable contains the index of best permutaion in 3d vector valid perm
  int bestPermIndex = -1;

  // Walk through all loops in a function in innermost-loop-first order.
  func.walk([&](AffineForOp fop) {
    // Finding the outermost loop
    Operation *par_op = fop.getParentOp();
    if (isa<FuncOp>(*par_op)) {
      // fop is the outermost loop
      bool if_present = checkif(fop);
      bool rectangularLoopNest = checkRect(fop);
      // This is true when there is no AffineIfOp inside the given loopnest and
      // also it is rectangular
      if (!(if_present) && rectangularLoopNest) {
        // Call to the function which will find all the dependences and access
        // matrices and their ranks inside loop body of fop
        find_dep_access(fop);

        // generating all valid permutations of loopnest and storing into
        // validperms
        valid_perm();

        if (validperms.size() > 1) {
          // This function finds the most efficient permutation
          bestPermIndex = best_perm();

          // This function permutes the loop for the best possible permutation
          bestLoopPermute(validperms[bestPermIndex], fop);
        }
      }
    }
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createAffineLoopInterchangePass() {
  return std::make_unique<LoopInterchange>();
}
