//===- IntegersSubsetMapping.h - Mapping subset ==> Successor ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// IntegersSubsetMapping is mapping from A to B, where
/// Items in A is subsets of integers,
/// Items in B some pointers (Successors).
/// If user which to add another subset for successor that is already
/// exists in mapping, IntegersSubsetMapping merges existing subset with
/// added one.
//
//===----------------------------------------------------------------------===//

#ifndef CRSBUILDER_H_
#define CRSBUILDER_H_

#include "llvm/Support/IntegersSubset.h"
#include <list>
#include <map>
#include <vector>

namespace llvm {

template <class SuccessorClass,
          class IntegersSubsetTy = IntegersSubset,
          class IntTy = IntItem>
class IntegersSubsetMapping {
  // FIXME: To much similar iterators typedefs, similar names. 
  //        - Rename RangeIterator to the cluster iterator.
  //        - Remove unused "add" methods.
  //        - Class contents needs cleaning.
public:
  
  typedef IntRange<IntTy> RangeTy;
  
  struct RangeEx : public RangeTy {
    RangeEx() : Weight(1) {}
    RangeEx(const RangeTy &R) : RangeTy(R), Weight(1) {}
    RangeEx(const RangeTy &R, unsigned W) : RangeTy(R), Weight(W) {}
    RangeEx(const IntTy &C) : RangeTy(C), Weight(1) {}
    RangeEx(const IntTy &L, const IntTy &H) : RangeTy(L, H), Weight(1) {}
    RangeEx(const IntTy &L, const IntTy &H, unsigned W) :
      RangeTy(L, H), Weight(W) {}
    unsigned Weight;
  };

  typedef std::pair<RangeEx, SuccessorClass*> Cluster;

  typedef std::list<RangeTy> RangesCollection;
  typedef typename RangesCollection::iterator RangesCollectionIt;
  typedef typename RangesCollection::const_iterator RangesCollectionConstIt;
  typedef IntegersSubsetMapping<SuccessorClass, IntegersSubsetTy, IntTy> self;
  
protected:

  typedef std::list<Cluster> CaseItems;
  typedef typename CaseItems::iterator CaseItemIt;
  typedef typename CaseItems::const_iterator CaseItemConstIt;
  
  // TODO: Change unclean CRS prefixes to SubsetMap for example.
  typedef std::map<SuccessorClass*, RangesCollection > CRSMap;
  typedef typename CRSMap::iterator CRSMapIt;

  struct ClustersCmp {
    bool operator()(const Cluster &C1, const Cluster &C2) {
      return C1.first < C2.first;
    }
  };
  
  CaseItems Items;
  bool Sorted;
  
  bool isIntersected(CaseItemIt& LItem, CaseItemIt& RItem) {
    return LItem->first.getHigh() >= RItem->first.getLow();
  }

  bool isJoinable(CaseItemIt& LItem, CaseItemIt& RItem) {
    if (LItem->second != RItem->second) {
      assert(!isIntersected(LItem, RItem) &&
             "Intersected items with different successors!");
      return false;
    }
    APInt RLow = RItem->first.getLow();
    if (RLow != APInt::getNullValue(RLow.getBitWidth()))
      --RLow;
    return LItem->first.getHigh() >= RLow;
  }
  
  void sort() {
    if (!Sorted) {
      std::vector<Cluster> clustersVector;
      clustersVector.reserve(Items.size());
      clustersVector.insert(clustersVector.begin(), Items.begin(), Items.end());
      std::sort(clustersVector.begin(), clustersVector.end(), ClustersCmp());
      Items.clear();
      Items.insert(Items.begin(), clustersVector.begin(), clustersVector.end());
      Sorted = true;
    }
  }
  
  enum DiffProcessState {
    L_OPENED,
    INTERSECT_OPENED,
    R_OPENED,
    ALL_IS_CLOSED
  };
  
  class DiffStateMachine {
    
    DiffProcessState State;
    IntTy OpenPt;
    SuccessorClass *CurrentLSuccessor;
    SuccessorClass *CurrentRSuccessor;
    
    self *LeftMapping;
    self *IntersectionMapping;
    self *RightMapping;

  public:
    
    typedef
      IntegersSubsetMapping<SuccessorClass, IntegersSubsetTy, IntTy> MappingTy;
    
    DiffStateMachine(MappingTy *L,
                                 MappingTy *Intersection,
                                 MappingTy *R) :
                                 State(ALL_IS_CLOSED),
                                 LeftMapping(L),
                                 IntersectionMapping(Intersection),
                                 RightMapping(R)
                                 {}
    
    void onLOpen(const IntTy &Pt, SuccessorClass *S) {
      switch (State) {
      case R_OPENED:
        if (Pt > OpenPt/*Don't add empty ranges.*/ && RightMapping) 
          RightMapping->add(OpenPt, Pt-1, CurrentRSuccessor);
        State = INTERSECT_OPENED;
        break;
      case ALL_IS_CLOSED:
        State = L_OPENED;
        break;
      default:
        assert(0 && "Got unexpected point.");
        break;
      }
      CurrentLSuccessor = S;
      OpenPt = Pt;
    }
    
    void onLClose(const IntTy &Pt) {
      switch (State) {
      case L_OPENED:
        assert(Pt >= OpenPt &&
               "Subset is not sorted or contains overlapped ranges");
        if (LeftMapping)
          LeftMapping->add(OpenPt, Pt, CurrentLSuccessor);
        State = ALL_IS_CLOSED;
        break;
      case INTERSECT_OPENED:
        if (IntersectionMapping)
          IntersectionMapping->add(OpenPt, Pt, CurrentLSuccessor);
        OpenPt = Pt + 1;
        State = R_OPENED;
        break;
      default:
        assert(0 && "Got unexpected point.");
        break;
      }
    }
    
    void onROpen(const IntTy &Pt, SuccessorClass *S) {
      switch (State) {
      case L_OPENED:
        if (Pt > OpenPt && LeftMapping)
          LeftMapping->add(OpenPt, Pt-1, CurrentLSuccessor);
        State = INTERSECT_OPENED;
        break;
      case ALL_IS_CLOSED:
        State = R_OPENED;
        break;
      default:
        assert(0 && "Got unexpected point.");
        break;
      }
      CurrentRSuccessor = S;
      OpenPt = Pt;      
    }
    
    void onRClose(const IntTy &Pt) {
      switch (State) {
      case R_OPENED:
        assert(Pt >= OpenPt &&
               "Subset is not sorted or contains overlapped ranges");
        if (RightMapping)
          RightMapping->add(OpenPt, Pt, CurrentRSuccessor);
        State = ALL_IS_CLOSED;
        break;
      case INTERSECT_OPENED:
        if (IntersectionMapping)
          IntersectionMapping->add(OpenPt, Pt, CurrentLSuccessor);
        OpenPt = Pt + 1;
        State = L_OPENED;
        break;
      default:
        assert(0 && "Got unexpected point.");
        break;
      }
    }
    
    void onLROpen(const IntTy &Pt,
                  SuccessorClass *LS,
                  SuccessorClass *RS) {
      switch (State) {
      case ALL_IS_CLOSED:
        State = INTERSECT_OPENED;
        break;
      default:
        assert(0 && "Got unexpected point.");
        break;
      }
      CurrentLSuccessor = LS;
      CurrentRSuccessor = RS;
      OpenPt = Pt;        
    }
    
    void onLRClose(const IntTy &Pt) {
      switch (State) {
      case INTERSECT_OPENED:
        if (IntersectionMapping)
          IntersectionMapping->add(OpenPt, Pt, CurrentLSuccessor);
        State = ALL_IS_CLOSED;
        break;
      default:
        assert(0 && "Got unexpected point.");
        break;        
      }
    }
    
    bool isLOpened() { return State == L_OPENED; }
    bool isROpened() { return State == R_OPENED; }
  };

public:
  
  // Don't public CaseItems itself. Don't allow edit the Items directly. 
  // Just present the user way to iterate over the internal collection
  // sharing iterator, begin() and end(). Editing should be controlled by
  // factory.
  typedef CaseItemIt RangeIterator;
  
  typedef std::pair<SuccessorClass*, IntegersSubsetTy> Case;
  typedef std::list<Case> Cases;
  typedef typename Cases::iterator CasesIt;
  
  IntegersSubsetMapping() {
    Sorted = false;
  }
  
  bool verify() {
    RangeIterator DummyErrItem;
    return verify(DummyErrItem);
  }
  
  bool verify(RangeIterator& errItem) {
    if (Items.empty())
      return true;
    sort();
    for (CaseItemIt j = Items.begin(), i = j++, e = Items.end();
         j != e; i = j++) {
      if (isIntersected(i, j) && i->second != j->second) {
        errItem = j;
        return false;
      }
    }
    return true;
  }

  bool isOverlapped(self &RHS) {
    if (Items.empty() || RHS.empty())
      return true;
    
    for (CaseItemIt L = Items.begin(), R = RHS.Items.begin(),
         el = Items.end(), er = RHS.Items.end(); L != el && R != er;) {
      
      const RangeTy &LRange = L->first;
      const RangeTy &RRange = R->first;
      
      if (LRange.getLow() > RRange.getLow()) {
        if (RRange.isSingleNumber() || LRange.getLow() > RRange.getHigh())
          ++R;
        else
          return true;
      } else if (LRange.getLow() < RRange.getLow()) {
        if (LRange.isSingleNumber() || LRange.getHigh() < RRange.getLow())
          ++L;
        else
          return true;
      } else // iRange.getLow() == jRange.getLow() 
        return true;
    }
    return false;
  }
   
  
  void optimize() {
    if (Items.size() < 2)
      return;
    sort();
    CaseItems OldItems = Items;
    Items.clear();
    const IntTy *Low = &OldItems.begin()->first.getLow();
    const IntTy *High = &OldItems.begin()->first.getHigh();
    unsigned Weight = OldItems.begin()->first.Weight;
    SuccessorClass *Successor = OldItems.begin()->second;
    for (CaseItemIt j = OldItems.begin(), i = j++, e = OldItems.end();
         j != e; i = j++) {
      if (isJoinable(i, j)) {
        const IntTy *CurHigh = &j->first.getHigh();
        Weight += j->first.Weight;
        if (*CurHigh > *High)
          High = CurHigh;
      } else {
        RangeEx R(*Low, *High, Weight);
        add(R, Successor);
        Low = &j->first.getLow();
        High = &j->first.getHigh(); 
        Weight = j->first.Weight;
        Successor = j->second;
      }
    }
    RangeEx R(*Low, *High, Weight);
    add(R, Successor);
    // We recollected the Items, but we kept it sorted.
    Sorted = true;
  }
  
  /// Adds a constant value.
  void add(const IntTy &C, SuccessorClass *S = 0) {
    RangeTy R(C);
    add(R, S);
  }
  
  /// Adds a range.
  void add(const IntTy &Low, const IntTy &High, SuccessorClass *S = 0) {
    RangeTy R(Low, High);
    add(R, S);
  }
  void add(const RangeTy &R, SuccessorClass *S = 0) {
    RangeEx REx = R;
    add(REx, S);
  }   
  void add(const RangeEx &R, SuccessorClass *S = 0) {
    Items.push_back(std::make_pair(R, S));
    Sorted = false;
  }  
  
  /// Adds all ranges and values from given ranges set to the current
  /// mapping.
  void add(const IntegersSubsetTy &CRS, SuccessorClass *S = 0,
           unsigned Weight = 0) {
    unsigned ItemWeight = 1;
    if (Weight)
      // Weight is associated with CRS, for now we perform a division to
      // get the weight for each item.
      ItemWeight = Weight / CRS.getNumItems();
    for (unsigned i = 0, e = CRS.getNumItems(); i < e; ++i) {
      RangeTy R = CRS.getItem(i);
      RangeEx REx(R, ItemWeight);
      add(REx, S);
    }
  }
  
  void add(self& RHS) {
    Items.insert(Items.end(), RHS.Items.begin(), RHS.Items.end());
  }
  
  void add(self& RHS, SuccessorClass *S) {
    for (CaseItemIt i = RHS.Items.begin(), e = RHS.Items.end(); i != e; ++i)
      add(i->first, S);
  }  
  
  void add(const RangesCollection& RHS, SuccessorClass *S = 0) {
    for (RangesCollectionConstIt i = RHS.begin(), e = RHS.end(); i != e; ++i)
      add(*i, S);
  }  
  
  /// Removes items from set.
  void removeItem(RangeIterator i) { Items.erase(i); }
  
  /// Moves whole case from current mapping to the NewMapping object.
  void detachCase(self& NewMapping, SuccessorClass *Succ) {
    for (CaseItemIt i = Items.begin(); i != Items.end();)
      if (i->second == Succ) {
        NewMapping.add(i->first, i->second);
        Items.erase(i++);
      } else
        ++i;
  }
  
  /// Removes all clusters for given successor.
  void removeCase(SuccessorClass *Succ) {
    for (CaseItemIt i = Items.begin(); i != Items.end();)
      if (i->second == Succ) {
        Items.erase(i++);
      } else
        ++i;
  }  
  
  /// Find successor that satisfies given value.
  SuccessorClass *findSuccessor(const IntTy& Val) {
    for (CaseItemIt i = Items.begin(); i != Items.end(); ++i) {
      if (i->first.isInRange(Val))
        return i->second;
    }
    return 0;
  }  
  
  /// Calculates the difference between this mapping and RHS.
  /// THIS without RHS is placed into LExclude,
  /// RHS without THIS is placed into RExclude,
  /// THIS intersect RHS is placed into Intersection.
  void diff(self *LExclude, self *Intersection, self *RExclude,
                             const self& RHS) {
    
    DiffStateMachine Machine(LExclude, Intersection, RExclude);
    
    CaseItemConstIt L = Items.begin(), R = RHS.Items.begin();
    while (L != Items.end() && R != RHS.Items.end()) {
      const Cluster &LCluster = *L;
      const RangeEx &LRange = LCluster.first;
      const Cluster &RCluster = *R;
      const RangeEx &RRange = RCluster.first;
      
      if (LRange.getHigh() < RRange.getLow()) {
        Machine.onLOpen(LRange.getLow(), LCluster.second);
        Machine.onLClose(LRange.getHigh());
        ++L;
        continue;
      }
      
      if (LRange.getLow() > RRange.getHigh()) {
        Machine.onROpen(RRange.getLow(), RCluster.second);
        Machine.onRClose(RRange.getHigh());
        ++R;
        continue;
      }

      if (LRange.getLow() < RRange.getLow()) {
        // May be opened in previous iteration.
        if (!Machine.isLOpened())
          Machine.onLOpen(LRange.getLow(), LCluster.second);
        Machine.onROpen(RRange.getLow(), RCluster.second);
      }
      else if (RRange.getLow() < LRange.getLow()) {
        if (!Machine.isROpened())
          Machine.onROpen(RRange.getLow(), RCluster.second);
        Machine.onLOpen(LRange.getLow(), LCluster.second);
      }
      else
        Machine.onLROpen(LRange.getLow(), LCluster.second, RCluster.second);
      
      if (LRange.getHigh() < RRange.getHigh()) {
        Machine.onLClose(LRange.getHigh());
        ++L;
        while(L != Items.end() && L->first.getHigh() < RRange.getHigh()) {
          Machine.onLOpen(L->first.getLow(), L->second);
          Machine.onLClose(L->first.getHigh());
          ++L;
        }
      }
      else if (RRange.getHigh() < LRange.getHigh()) {
        Machine.onRClose(RRange.getHigh());
        ++R;
        while(R != RHS.Items.end() && R->first.getHigh() < LRange.getHigh()) {
          Machine.onROpen(R->first.getLow(), R->second);
          Machine.onRClose(R->first.getHigh());
          ++R;
        }
      }
      else {
        Machine.onLRClose(LRange.getHigh());
        ++L;
        ++R;
      }
    }
    
    if (L != Items.end()) {
      if (Machine.isLOpened()) {
        Machine.onLClose(L->first.getHigh());
        ++L;
      }
      if (LExclude)
        while (L != Items.end()) {
          LExclude->add(L->first, L->second);
          ++L;
        }
    } else if (R != RHS.Items.end()) {
      if (Machine.isROpened()) {
        Machine.onRClose(R->first.getHigh());
        ++R;
      }
      if (RExclude)
        while (R != RHS.Items.end()) {
          RExclude->add(R->first, R->second);
          ++R;
        }
    }
  }  
  
  /// Builds the finalized case objects.
  void getCases(Cases& TheCases, bool PreventMerging = false) {
    //FIXME: PreventMerging is a temporary parameter.
    //Currently a set of passes is still knows nothing about
    //switches with case ranges, and if these passes meet switch
    //with complex case that crashs the application.
    if (PreventMerging) {
      for (RangeIterator i = this->begin(); i != this->end(); ++i) {
        RangesCollection SingleRange;
        SingleRange.push_back(i->first);
        TheCases.push_back(std::make_pair(i->second,
                                          IntegersSubsetTy(SingleRange)));
      }
      return;
    }
    CRSMap TheCRSMap;
    for (RangeIterator i = this->begin(); i != this->end(); ++i)
      TheCRSMap[i->second].push_back(i->first);
    for (CRSMapIt i = TheCRSMap.begin(), e = TheCRSMap.end(); i != e; ++i)
      TheCases.push_back(std::make_pair(i->first, IntegersSubsetTy(i->second)));
  }
  
  /// Builds the finalized case objects ignoring successor values, as though
  /// all ranges belongs to the same successor.
  IntegersSubsetTy getCase() {
    RangesCollection Ranges;
    for (RangeIterator i = this->begin(); i != this->end(); ++i)
      Ranges.push_back(i->first);
    return IntegersSubsetTy(Ranges);
  }  
  
  /// Returns pointer to value of case if it is single-numbered or 0
  /// in another case.
  const IntTy* getCaseSingleNumber(SuccessorClass *Succ) {
    const IntTy* Res = 0;
    for (CaseItemIt i = Items.begin(); i != Items.end(); ++i)
      if (i->second == Succ) {
        if (!i->first.isSingleNumber())
          return 0;
        if (Res)
          return 0;
        else 
          Res = &(i->first.getLow());
      }
    return Res;
  }  
  
  /// Returns true if there is no ranges and values inside.
  bool empty() const { return Items.empty(); }
  
  void clear() {
    Items.clear();
    // Don't reset Sorted flag:
    // 1. For empty mapping it matters nothing.
    // 2. After first item will added Sorted flag will cleared.
  }  
  
  // Returns number of clusters
  unsigned size() const {
    return Items.size();
  }
  
  RangeIterator begin() { return Items.begin(); }
  RangeIterator end() { return Items.end(); }
};

class BasicBlock;
typedef IntegersSubsetMapping<BasicBlock> IntegersSubsetToBB;

}

#endif /* CRSBUILDER_H_ */
