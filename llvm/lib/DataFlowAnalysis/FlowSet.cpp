#include "llvm/DataFlowAnalysis/FlowSet.h"

using namespace llvm;

template <typename T>
typename FlowSet<T>::Node* FlowSet<T>::m_spRoot = nullptr;

template <typename T>
void FlowSet<T>::push_back(T data)
{
    Node* pTemp = getNode(data);
    if(!getRoot())
    {
	getRoot() = pTemp;
    }
    else
    {
	Node* pC = getRoot();
	while(pC->pNext)
	{
	    pC = pC->pNext;
	}
	pC->pNext = pTemp;
    }
}
