//===---------------- PBQP.cpp --------- PBQP Solver ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Developed by:                   Bernhard Scholz
//                             The University of Sydney
//                         http://www.it.usyd.edu.au/~scholz
//===----------------------------------------------------------------------===//

#include "PBQP.h"
#include "llvm/Config/alloca.h"
#include <limits>
#include <cassert>
#include <cstring>

namespace llvm {

/**************************************************************************
 * Data Structures 
 **************************************************************************/

/* edge of PBQP graph */
typedef struct adjnode {
  struct adjnode *prev,      /* doubly chained list */ 
                 *succ, 
                 *reverse;   /* reverse edge */
  int adj;                   /* adj. node */
  PBQPMatrix *costs;         /* cost matrix of edge */

  bool tc_valid;              /* flag whether following fields are valid */
  int *tc_safe_regs;          /* safe registers */
  int tc_impact;              /* impact */ 
} adjnode;

/* bucket node */
typedef struct bucketnode {
  struct bucketnode *prev;   /* doubly chained list */
  struct bucketnode *succ;   
  int u;                     /* node */
} bucketnode;

/* data structure of partitioned boolean quadratic problem */
struct pbqp {
  int num_nodes;             /* number of nodes */
  int max_deg;               /* maximal degree of a node */
  bool solved;               /* flag that indicates whether PBQP has been solved yet */
  bool optimal;              /* flag that indicates whether PBQP is optimal */
  PBQPNum min;
  bool changed;              /* flag whether graph has changed in simplification */

                             /* node fields */
  PBQPVector **node_costs;   /* cost vectors of nodes */
  int *node_deg;             /* node degree of nodes */
  int *solution;             /* solution for node */
  adjnode **adj_list;        /* adj. list */
  bucketnode **bucket_ptr;   /* bucket pointer of a node */

                             /* node stack */
  int *stack;                /* stack of nodes */
  int stack_ptr;             /* stack pointer */

                             /* bucket fields */
  bucketnode **bucket_list;  /* bucket list */

  int num_r0;                /* counters for number statistics */
  int num_ri;
  int num_rii;
  int num_rn; 
  int num_rn_special;      
};

bool isInf(PBQPNum n) { return n == std::numeric_limits<PBQPNum>::infinity(); } 

/*****************************************************************************
 * allocation/de-allocation of pbqp problem 
 ****************************************************************************/

/* allocate new partitioned boolean quadratic program problem */
pbqp *alloc_pbqp(int num_nodes)
{
  pbqp *this_;
  int u;
  
  assert(num_nodes > 0);
  
  /* allocate memory for pbqp data structure */   
  this_ = (pbqp *)malloc(sizeof(pbqp));

  /* Initialize pbqp fields */
  this_->num_nodes = num_nodes;
  this_->solved = false;
  this_->optimal = true;
  this_->min = 0.0;
  this_->max_deg = 0;
  this_->changed = false;
  this_->num_r0 = 0;
  this_->num_ri = 0;
  this_->num_rii = 0;
  this_->num_rn = 0;
  this_->num_rn_special = 0;
  
  /* initialize/allocate stack fields of pbqp */ 
  this_->stack = (int *) malloc(sizeof(int)*num_nodes);
  this_->stack_ptr = 0;
  
  /* initialize/allocate node fields of pbqp */
  this_->adj_list = (adjnode **) malloc(sizeof(adjnode *)*num_nodes);
  this_->node_deg = (int *) malloc(sizeof(int)*num_nodes);
  this_->solution = (int *) malloc(sizeof(int)*num_nodes);
  this_->bucket_ptr = (bucketnode **) malloc(sizeof(bucketnode **)*num_nodes);
  this_->node_costs = (PBQPVector**) malloc(sizeof(PBQPVector*) * num_nodes);
  for(u=0;u<num_nodes;u++) {
    this_->solution[u]=-1;
    this_->adj_list[u]=NULL;
    this_->node_deg[u]=0;
    this_->bucket_ptr[u]=NULL;
    this_->node_costs[u]=NULL;
  }
  
  /* initialize bucket list */
  this_->bucket_list = NULL;
  
  return this_;
}

/* free pbqp problem */
void free_pbqp(pbqp *this_)
{
  int u;
  int deg;
  adjnode *adj_ptr,*adj_next;
  bucketnode *bucket,*bucket_next;
  
  assert(this_ != NULL);
  
  /* free node cost fields */
  for(u=0;u < this_->num_nodes;u++) {
    delete this_->node_costs[u];
  }
  free(this_->node_costs);
  
  /* free bucket list */
  for(deg=0;deg<=this_->max_deg;deg++) {
    for(bucket=this_->bucket_list[deg];bucket!=NULL;bucket=bucket_next) {
      this_->bucket_ptr[bucket->u] = NULL;
      bucket_next = bucket-> succ;
      free(bucket);
    }
  }
  free(this_->bucket_list);
  
  /* free adj. list */
  assert(this_->adj_list != NULL);
  for(u=0;u < this_->num_nodes; u++) {
    for(adj_ptr = this_->adj_list[u]; adj_ptr != NULL; adj_ptr = adj_next) {
      adj_next = adj_ptr -> succ;
      if (u < adj_ptr->adj) {
        assert(adj_ptr != NULL);
        delete adj_ptr->costs;
      }
      if (adj_ptr -> tc_safe_regs != NULL) {
           free(adj_ptr -> tc_safe_regs);
      }
      free(adj_ptr);
    }
  }
  free(this_->adj_list);
  
  /* free other node fields */
  free(this_->node_deg);
  free(this_->solution);
  free(this_->bucket_ptr);

  /* free stack */
  free(this_->stack);

  /* free pbqp data structure itself */
  free(this_);
}


/****************************************************************************
 * adj. node routines 
 ****************************************************************************/

/* find data structure of adj. node of a given node */
static
adjnode *find_adjnode(pbqp *this_,int u,int v)
{
  adjnode *adj_ptr;
  
  assert (this_ != NULL);
  assert (u >= 0 && u < this_->num_nodes);
  assert (v >= 0 && v < this_->num_nodes);
  assert(this_->adj_list != NULL);

  for(adj_ptr = this_ -> adj_list[u];adj_ptr != NULL; adj_ptr = adj_ptr -> succ) {
    if (adj_ptr->adj == v) {
      return adj_ptr;
    }
  }
  return NULL;
}

/* allocate a new data structure for adj. node */
static
adjnode *alloc_adjnode(pbqp *this_,int u, PBQPMatrix *costs)
{
  adjnode *p;

  assert(this_ != NULL);
  assert(costs != NULL);
  assert(u >= 0 && u < this_->num_nodes);

  p = (adjnode *)malloc(sizeof(adjnode));
  assert(p != NULL);
  
  p->adj = u;
  p->costs = costs;  

  p->tc_valid= false;
  p->tc_safe_regs = NULL;
  p->tc_impact = 0;

  return p;
}

/* insert adjacence node to adj. list */
static
void insert_adjnode(pbqp *this_, int u, adjnode *adj_ptr)
{

  assert(this_ != NULL);
  assert(adj_ptr != NULL);
  assert(u >= 0 && u < this_->num_nodes);

  /* if adjacency list of node is not empty -> update
     first node of the list */
  if (this_ -> adj_list[u] != NULL) {
    assert(this_->adj_list[u]->prev == NULL);
    this_->adj_list[u] -> prev = adj_ptr;
  }

  /* update doubly chained list pointers of pointers */
  adj_ptr -> succ = this_->adj_list[u];
  adj_ptr -> prev = NULL;

  /* update adjacency list pointer of node u */
  this_->adj_list[u] = adj_ptr;
}

/* remove entry in an adj. list */
static
void remove_adjnode(pbqp *this_, int u, adjnode *adj_ptr)
{
  assert(this_!= NULL);
  assert(u >= 0 && u <= this_->num_nodes);
  assert(this_->adj_list != NULL);
  assert(adj_ptr != NULL);
  
  if (adj_ptr -> prev == NULL) {
    this_->adj_list[u] = adj_ptr -> succ;
  } else {
    adj_ptr -> prev -> succ = adj_ptr -> succ;
  } 

  if (adj_ptr -> succ != NULL) {
    adj_ptr -> succ -> prev = adj_ptr -> prev;
  }

  if(adj_ptr->reverse != NULL) {
    adjnode *rev = adj_ptr->reverse;
    rev->reverse = NULL;
  }

  if (adj_ptr -> tc_safe_regs != NULL) {
     free(adj_ptr -> tc_safe_regs);
  }

  free(adj_ptr);
}

/*****************************************************************************
 * node functions 
 ****************************************************************************/

/* get degree of a node */
static
int get_deg(pbqp *this_,int u)
{
  adjnode *adj_ptr;
  int deg = 0;
  
  assert(this_ != NULL);
  assert(u >= 0 && u < this_->num_nodes);
  assert(this_->adj_list != NULL);

  for(adj_ptr = this_ -> adj_list[u];adj_ptr != NULL; adj_ptr = adj_ptr -> succ) {
    deg ++;
  }
  return deg;
}

/* reinsert node */
static
void reinsert_node(pbqp *this_,int u)
{
  adjnode *adj_u,
          *adj_v;

  assert(this_!= NULL);
  assert(u >= 0 && u <= this_->num_nodes);
  assert(this_->adj_list != NULL);

  for(adj_u = this_ -> adj_list[u]; adj_u != NULL; adj_u = adj_u -> succ) {
    int v = adj_u -> adj;
    adj_v = alloc_adjnode(this_,u,adj_u->costs);
    insert_adjnode(this_,v,adj_v);
  }
}

/* remove node */
static
void remove_node(pbqp *this_,int u)
{
  adjnode *adj_ptr;

  assert(this_!= NULL);
  assert(u >= 0 && u <= this_->num_nodes);
  assert(this_->adj_list != NULL);

  for(adj_ptr = this_ -> adj_list[u]; adj_ptr != NULL; adj_ptr = adj_ptr -> succ) {
    remove_adjnode(this_,adj_ptr->adj,adj_ptr -> reverse);
  }
}

/*****************************************************************************
 * edge functions
 ****************************************************************************/

/* insert edge to graph */
/* (does not check whether edge exists in graph */
static
void insert_edge(pbqp *this_, int u, int v, PBQPMatrix *costs)
{
  adjnode *adj_u,
          *adj_v;
  
  /* create adjanceny entry for u */
  adj_u = alloc_adjnode(this_,v,costs);
  insert_adjnode(this_,u,adj_u);


  /* create adjanceny entry for v */
  adj_v = alloc_adjnode(this_,u,costs);
  insert_adjnode(this_,v,adj_v);
  
  /* create link for reverse edge */
  adj_u -> reverse = adj_v;
  adj_v -> reverse = adj_u;
}

/* delete edge */
static
void delete_edge(pbqp *this_,int u,int v)
{
  adjnode *adj_ptr;
  adjnode *rev;
  
  assert(this_ != NULL);
  assert( u >= 0 && u < this_->num_nodes);
  assert( v >= 0 && v < this_->num_nodes);

  adj_ptr=find_adjnode(this_,u,v);
  assert(adj_ptr != NULL);
  assert(adj_ptr->reverse != NULL);

  delete adj_ptr -> costs;
 
  rev = adj_ptr->reverse; 
  remove_adjnode(this_,u,adj_ptr);
  remove_adjnode(this_,v,rev);
} 

/*****************************************************************************
 * cost functions 
 ****************************************************************************/

/* Note: Since cost(u,v) = transpose(cost(v,u)), it would be necessary to store 
   two matrices for both edges (u,v) and (v,u). However, we only store the 
   matrix for the case u < v. For the other case we transpose the stored matrix
   if required. 
*/

/* add costs to cost vector of a node */
void add_pbqp_nodecosts(pbqp *this_,int u, PBQPVector *costs)
{
  assert(this_ != NULL);
  assert(costs != NULL);
  assert(u >= 0 && u <= this_->num_nodes);
  
  if (!this_->node_costs[u]) {
    this_->node_costs[u] = new PBQPVector(*costs);
  } else {
    *this_->node_costs[u] += *costs;
  }
}

/* get cost matrix ptr */
static
PBQPMatrix *get_costmatrix_ptr(pbqp *this_, int u, int v)
{
  adjnode *adj_ptr;
  PBQPMatrix *m = NULL;

  assert (this_ != NULL);
  assert (u >= 0 && u < this_->num_nodes);
  assert (v >= 0 && v < this_->num_nodes); 

  adj_ptr = find_adjnode(this_,u,v);

  if (adj_ptr != NULL) {
    m = adj_ptr -> costs;
  } 

  return m;
}

/* get cost matrix ptr */
/* Note: only the pointer is returned for 
   cost(u,v), if u < v.
*/ 
static
PBQPMatrix *pbqp_get_costmatrix(pbqp *this_, int u, int v)
{
  adjnode *adj_ptr = find_adjnode(this_,u,v);
  
  if (adj_ptr != NULL) {
    if ( u < v) {
      return new PBQPMatrix(*adj_ptr->costs);
    } else {
      return new PBQPMatrix(adj_ptr->costs->transpose());
    }
  } else {
    return NULL;
  }  
}

/* add costs to cost matrix of an edge */
void add_pbqp_edgecosts(pbqp *this_,int u,int v, PBQPMatrix *costs)
{
  PBQPMatrix *adj_costs;

  assert(this_!= NULL);
  assert(costs != NULL);
  assert(u >= 0 && u <= this_->num_nodes);
  assert(v >= 0 && v <= this_->num_nodes);
  
  /* does the edge u-v exists ? */
  if (u == v) {
    PBQPVector *diag = new PBQPVector(costs->diagonalize());
    add_pbqp_nodecosts(this_,v,diag);
    delete diag;
  } else if ((adj_costs = get_costmatrix_ptr(this_,u,v))!=NULL) {
    if ( u < v) {
      *adj_costs += *costs;
    } else {
      *adj_costs += costs->transpose();
    }
  } else {
    adj_costs = new PBQPMatrix((u < v) ? *costs : costs->transpose());
    insert_edge(this_,u,v,adj_costs);
  } 
}

/* remove bucket from bucket list */
static
void pbqp_remove_bucket(pbqp *this_, bucketnode *bucket)
{
  int u = bucket->u;
  
  assert(this_ != NULL);
  assert(u >= 0 && u < this_->num_nodes);
  assert(this_->bucket_list != NULL);
  assert(this_->bucket_ptr[u] != NULL);
  
  /* update predecessor node in bucket list 
     (if no preceeding bucket exists, then
     the bucket_list pointer needs to be 
     updated.)
  */    
  if (bucket->prev != NULL) {
    bucket->prev-> succ = bucket->succ; 
  } else {
    this_->bucket_list[this_->node_deg[u]] = bucket -> succ;
  }
  
  /* update successor node in bucket list */ 
  if (bucket->succ != NULL) { 
    bucket->succ-> prev = bucket->prev;
  }
}

/**********************************************************************************
 * pop functions
 **********************************************************************************/

/* pop node of given degree */
static
int pop_node(pbqp *this_,int deg)
{
  bucketnode *bucket;
  int u;

  assert(this_ != NULL);
  assert(deg >= 0 && deg <= this_->max_deg);
  assert(this_->bucket_list != NULL);
   
  /* get first bucket of bucket list */
  bucket = this_->bucket_list[deg];
  assert(bucket != NULL);

  /* remove bucket */
  pbqp_remove_bucket(this_,bucket);
  u = bucket->u;
  free(bucket);
  return u;
}

/**********************************************************************************
 * reorder functions
 **********************************************************************************/

/* add bucket to bucketlist */
static
void add_to_bucketlist(pbqp *this_,bucketnode *bucket, int deg)
{
  bucketnode *old_head;
  
  assert(bucket != NULL);
  assert(this_ != NULL);
  assert(deg >= 0 && deg <= this_->max_deg);
  assert(this_->bucket_list != NULL);

  /* store node degree (for re-ordering purposes)*/
  this_->node_deg[bucket->u] = deg;
  
  /* put bucket to front of doubly chained list */
  old_head = this_->bucket_list[deg];
  bucket -> prev = NULL;
  bucket -> succ = old_head;
  this_ -> bucket_list[deg] = bucket;
  if (bucket -> succ != NULL ) {
    assert ( old_head -> prev == NULL);
    old_head -> prev = bucket;
  }
}


/* reorder node in bucket list according to 
   current node degree */
static
void reorder_node(pbqp *this_, int u)
{
  int deg; 
  
  assert(this_ != NULL);
  assert(u>= 0 && u < this_->num_nodes);
  assert(this_->bucket_list != NULL);
  assert(this_->bucket_ptr[u] != NULL);

  /* get current node degree */
  deg = get_deg(this_,u);
  
  /* remove bucket from old bucket list only
     if degree of node has changed. */
  if (deg != this_->node_deg[u]) {
    pbqp_remove_bucket(this_,this_->bucket_ptr[u]);
    add_to_bucketlist(this_,this_->bucket_ptr[u],deg);
  } 
}

/* reorder adj. nodes of a node */
static
void reorder_adjnodes(pbqp *this_,int u)
{
  adjnode *adj_ptr;
  
  assert(this_!= NULL);
  assert(u >= 0 && u <= this_->num_nodes);
  assert(this_->adj_list != NULL);

  for(adj_ptr = this_ -> adj_list[u]; adj_ptr != NULL; adj_ptr = adj_ptr -> succ) {
    reorder_node(this_,adj_ptr->adj);
  }
}

/**********************************************************************************
 * creation functions
 **********************************************************************************/

/* create new bucket entry */
/* consistency of the bucket list is not checked! */
static
void create_bucket(pbqp *this_,int u,int deg)
{
  bucketnode *bucket;
  
  assert(this_ != NULL);
  assert(u >= 0 && u < this_->num_nodes);
  assert(this_->bucket_list != NULL);
  
  bucket = (bucketnode *)malloc(sizeof(bucketnode));
  assert(bucket != NULL);

  bucket -> u = u;
  this_->bucket_ptr[u] = bucket;

  add_to_bucketlist(this_,bucket,deg);
}

/* create bucket list */
static
void create_bucketlist(pbqp *this_)
{
  int u;
  int max_deg;
  int deg;

  assert(this_ != NULL);
  assert(this_->bucket_list == NULL);

  /* determine max. degree of the nodes */
  max_deg = 2;  /* at least of degree two! */
  for(u=0;u<this_->num_nodes;u++) {
    deg = this_->node_deg[u] = get_deg(this_,u);
    if (deg > max_deg) {
      max_deg = deg;
    }
  }
  this_->max_deg = max_deg;
  
  /* allocate bucket list */
  this_ -> bucket_list = (bucketnode **)malloc(sizeof(bucketnode *)*(max_deg + 1));
  memset(this_->bucket_list,0,sizeof(bucketnode *)*(max_deg + 1));
  assert(this_->bucket_list != NULL);
  
  /* insert nodes to the list */
  for(u=0;u<this_->num_nodes;u++) {
    create_bucket(this_,u,this_->node_deg[u]);  
  }
}

/*****************************************************************************
 * PBQP simplification for trivial nodes
 ****************************************************************************/

/* remove trivial node with cost vector length of one */
static
void disconnect_trivialnode(pbqp *this_,int u)
{
  int v;
  adjnode *adj_ptr, 
          *next;
  PBQPMatrix *c_uv;
  PBQPVector *c_v;
  
  assert(this_ != NULL);
  assert(this_->node_costs != NULL);
  assert(u >= 0 && u < this_ -> num_nodes);
  assert(this_->node_costs[u]->getLength() == 1);
  
  /* add edge costs to node costs of adj. nodes */
  for(adj_ptr = this_->adj_list[u]; adj_ptr != NULL; adj_ptr = next){
    next = adj_ptr -> succ;
    v = adj_ptr -> adj;
    assert(v >= 0 && v < this_ -> num_nodes);
    
    /* convert matrix to cost vector offset for adj. node */
    c_uv = pbqp_get_costmatrix(this_,u,v);
    c_v = new PBQPVector(c_uv->getRowAsVector(0));
    *this_->node_costs[v] += *c_v;
    
    /* delete edge & free vec/mat */
    delete c_v;
    delete c_uv;
    delete_edge(this_,u,v);
  }   
}

/* find all trivial nodes and disconnect them */
static   
void eliminate_trivial_nodes(pbqp *this_)
{
   int u;
   
   assert(this_ != NULL);
   assert(this_ -> node_costs != NULL);
   
   for(u=0;u < this_ -> num_nodes; u++) {
     if (this_->node_costs[u]->getLength() == 1) {
       disconnect_trivialnode(this_,u); 
     }
   }
}

/*****************************************************************************
 * Normal form for PBQP 
 ****************************************************************************/

/* simplify a cost matrix. If the matrix
   is independent, then simplify_matrix
   returns true - otherwise false. In
   vectors u and v the offset values of
   the decomposition are stored. 
*/

static
bool normalize_matrix(PBQPMatrix *m, PBQPVector *u, PBQPVector *v)
{
  assert( m != NULL);
  assert( u != NULL);
  assert( v != NULL);
  assert( u->getLength() > 0);
  assert( v->getLength() > 0);
  
  assert(m->getRows() == u->getLength());
  assert(m->getCols() == v->getLength());

  /* determine u vector */
  for(unsigned r = 0; r < m->getRows(); ++r) {
    PBQPNum min = m->getRowMin(r);
    (*u)[r] += min;
    if (!isInf(min)) {
      m->subFromRow(r, min);
    } else {
      m->setRow(r, 0);
    }
  }
  
  /* determine v vector */
  for(unsigned c = 0; c < m->getCols(); ++c) {
    PBQPNum min = m->getColMin(c);
    (*v)[c] += min;
    if (!isInf(min)) {
      m->subFromCol(c, min);
    } else {
      m->setCol(c, 0);
    }
  }
  
  /* determine whether matrix is 
     independent or not. 
    */
  return m->isZero();
}

/* simplify single edge */
static
void simplify_edge(pbqp *this_,int u,int v)
{
  PBQPMatrix *costs;
  bool is_zero; 
  
  assert (this_ != NULL);
  assert (u >= 0 && u <this_->num_nodes);
  assert (v >= 0 && v <this_->num_nodes);
  assert (u != v);

  /* swap u and v  if u > v in order to avoid un-necessary
     tranpositions of the cost matrix */
  
  if (u > v) {
    int swap = u;
    u = v;
    v = swap;
  }
  
  /* get cost matrix and simplify it */  
  costs = get_costmatrix_ptr(this_,u,v);
  is_zero=normalize_matrix(costs,this_->node_costs[u],this_->node_costs[v]);

  /* delete edge */
  if(is_zero){
    delete_edge(this_,u,v);
    this_->changed = true;
  }
}

/* normalize cost matrices and remove 
   edges in PBQP if they ary independent, 
   i.e. can be decomposed into two 
   cost vectors.
*/
static
void eliminate_independent_edges(pbqp *this_)
{
  int u,v;
  adjnode *adj_ptr,*next;
  
  assert(this_ != NULL);
  assert(this_ -> adj_list != NULL);

  this_->changed = false;
  for(u=0;u < this_->num_nodes;u++) {
    for (adj_ptr = this_ -> adj_list[u]; adj_ptr != NULL; adj_ptr = next) {
      next = adj_ptr -> succ;
      v = adj_ptr -> adj;
      assert(v >= 0 && v < this_->num_nodes);
      if (u < v) {
        simplify_edge(this_,u,v);
      } 
    }
  }
}


/*****************************************************************************
 * PBQP reduction rules 
 ****************************************************************************/

/* RI reduction
   This reduction rule is applied for nodes 
   of degree one. */

static
void apply_RI(pbqp *this_,int x)
{
  int y;
  unsigned xlen,
           ylen;
  PBQPMatrix *c_yx;
  PBQPVector *c_x, *delta;
  
  assert(this_ != NULL);
  assert(x >= 0 && x < this_->num_nodes);
  assert(this_ -> adj_list[x] != NULL);
  assert(this_ -> adj_list[x] -> succ == NULL);

  /* get adjacence matrix */
  y = this_ -> adj_list[x] -> adj;
  assert(y >= 0 && y < this_->num_nodes);
  
  /* determine length of cost vectors for node x and y */
  xlen = this_ -> node_costs[x]->getLength();
  ylen = this_ -> node_costs[y]->getLength();

  /* get cost vector c_x and matrix c_yx */
  c_x = this_ -> node_costs[x];
  c_yx = pbqp_get_costmatrix(this_,y,x); 
  assert (c_yx != NULL);

  
  /* allocate delta vector */
  delta = new PBQPVector(ylen);

  /* compute delta vector */
  for(unsigned i = 0; i < ylen; ++i) {
    PBQPNum min =  (*c_yx)[i][0] + (*c_x)[0];
    for(unsigned j = 1; j < xlen; ++j) {
      PBQPNum c =  (*c_yx)[i][j] + (*c_x)[j];
      if ( c < min )  
         min = c;
    }
    (*delta)[i] = min; 
  } 

  /* add delta vector */
  *this_ -> node_costs[y] += *delta;

  /* delete node x */
  remove_node(this_,x);

  /* reorder adj. nodes of node x */
  reorder_adjnodes(this_,x);

  /* push node x on stack */
  assert(this_ -> stack_ptr < this_ -> num_nodes);
  this_->stack[this_ -> stack_ptr++] = x;

  /* free vec/mat */
  delete c_yx;
  delete delta;

  /* increment counter for number statistic */
  this_->num_ri++;
}

/* RII reduction
   This reduction rule is applied for nodes 
   of degree two. */

static
void apply_RII(pbqp *this_,int x)
{
  int y,z; 
  unsigned xlen,ylen,zlen;
  adjnode *adj_yz;

  PBQPMatrix *c_yx, *c_zx;
  PBQPVector *cx;
  PBQPMatrix *delta;
 
  assert(this_ != NULL);
  assert(x >= 0 && x < this_->num_nodes);
  assert(this_ -> adj_list[x] != NULL);
  assert(this_ -> adj_list[x] -> succ != NULL);
  assert(this_ -> adj_list[x] -> succ -> succ == NULL);

  /* get adjacence matrix */
  y = this_ -> adj_list[x] -> adj;
  z = this_ -> adj_list[x] -> succ -> adj;
  assert(y >= 0 && y < this_->num_nodes);
  assert(z >= 0 && z < this_->num_nodes);
  
  /* determine length of cost vectors for node x and y */
  xlen = this_ -> node_costs[x]->getLength();
  ylen = this_ -> node_costs[y]->getLength();
  zlen = this_ -> node_costs[z]->getLength();

  /* get cost vector c_x and matrix c_yx */
  cx = this_ -> node_costs[x];
  c_yx = pbqp_get_costmatrix(this_,y,x); 
  c_zx = pbqp_get_costmatrix(this_,z,x); 
  assert(c_yx != NULL);
  assert(c_zx != NULL);

  /* Colour Heuristic */
  if ( (adj_yz = find_adjnode(this_,y,z)) != NULL) {
    adj_yz->tc_valid = false;
    adj_yz->reverse->tc_valid = false; 
  }

  /* allocate delta matrix */
  delta = new PBQPMatrix(ylen, zlen);

  /* compute delta matrix */
  for(unsigned i=0;i<ylen;i++) {
    for(unsigned j=0;j<zlen;j++) {
      PBQPNum min = (*c_yx)[i][0] + (*c_zx)[j][0] + (*cx)[0];
      for(unsigned k=1;k<xlen;k++) {
        PBQPNum c = (*c_yx)[i][k] + (*c_zx)[j][k] + (*cx)[k];
        if ( c < min ) {
          min = c;
        }
      }
      (*delta)[i][j] = min;
    }
  }

  /* add delta matrix */
  add_pbqp_edgecosts(this_,y,z,delta);

  /* delete node x */
  remove_node(this_,x);

  /* simplify cost matrix c_yz */
  simplify_edge(this_,y,z);

  /* reorder adj. nodes */
  reorder_adjnodes(this_,x);

  /* push node x on stack */
  assert(this_ -> stack_ptr < this_ -> num_nodes);
  this_->stack[this_ -> stack_ptr++] = x;

  /* free vec/mat */
  delete c_yx;
  delete c_zx;
  delete delta;

  /* increment counter for number statistic */
  this_->num_rii++;

}

/* RN reduction */
static
void apply_RN(pbqp *this_,int x)
{
  unsigned xlen;

  assert(this_ != NULL);
  assert(x >= 0 && x < this_->num_nodes);
  assert(this_ -> node_costs[x] != NULL);

  xlen = this_ -> node_costs[x] -> getLength();

  /* after application of RN rule no optimality
     can be guaranteed! */
  this_ -> optimal = false;
  
  /* push node x on stack */
  assert(this_ -> stack_ptr < this_ -> num_nodes);
  this_->stack[this_ -> stack_ptr++] = x;

  /* delete node x */ 
  remove_node(this_,x);

  /* reorder adj. nodes of node x */
  reorder_adjnodes(this_,x);

  /* increment counter for number statistic */
  this_->num_rn++;
}


static
void compute_tc_info(pbqp *this_, adjnode *p)
{
   adjnode *r;
   PBQPMatrix *m;
   int x,y;
   PBQPVector *c_x, *c_y;
   int *row_inf_counts;

   assert(p->reverse != NULL);

   /* set flags */ 
   r = p->reverse;
   p->tc_valid = true;
   r->tc_valid = true;

   /* get edge */
   x = r->adj;
   y = p->adj;

   /* get cost vectors */
   c_x = this_ -> node_costs[x];
   c_y = this_ -> node_costs[y];

   /* get cost matrix */
   m = pbqp_get_costmatrix(this_, x, y);

  
   /* allocate allowed set for edge (x,y) and (y,x) */
   if (p->tc_safe_regs == NULL) {
     p->tc_safe_regs = (int *) malloc(sizeof(int) * c_x->getLength());
   } 

   if (r->tc_safe_regs == NULL ) {
     r->tc_safe_regs = (int *) malloc(sizeof(int) * c_y->getLength());
   }

   p->tc_impact = r->tc_impact = 0;

   row_inf_counts = (int *) alloca(sizeof(int) * c_x->getLength());

   /* init arrays */
   p->tc_safe_regs[0] = 0;
   row_inf_counts[0] = 0;
   for(unsigned i = 1; i < c_x->getLength(); ++i){
     p->tc_safe_regs[i] = 1;
     row_inf_counts[i] = 0;
   }

   r->tc_safe_regs[0] = 0;
   for(unsigned j = 1; j < c_y->getLength(); ++j){
     r->tc_safe_regs[j] = 1;
   }

   for(unsigned j = 0; j < c_y->getLength(); ++j) {
      int col_inf_counts = 0;
      for (unsigned i = 0; i < c_x->getLength(); ++i) {
         if (isInf((*m)[i][j])) {
              ++col_inf_counts;
              ++row_inf_counts[i];
         
              p->tc_safe_regs[i] = 0;
              r->tc_safe_regs[j] = 0;
         }
      }
      if (col_inf_counts > p->tc_impact) {
           p->tc_impact = col_inf_counts;
      }
   }

   for(unsigned i = 0; i < c_x->getLength(); ++i){
     if (row_inf_counts[i] > r->tc_impact)
     {
           r->tc_impact = row_inf_counts[i];
     }
   }
           
   delete m;
}

/* 
 * Checks whether node x can be locally coloured. 
 */
static 
int is_colorable(pbqp *this_,int x)
{
  adjnode *adj_ptr;
  PBQPVector *c_x;
  int result = 1;
  int *allowed;
  int num_allowed = 0;
  unsigned total_impact = 0;

  assert(this_ != NULL);
  assert(x >= 0 && x < this_->num_nodes);
  assert(this_ -> node_costs[x] != NULL);

  c_x = this_ -> node_costs[x];

  /* allocate allowed set */
  allowed = (int *)malloc(sizeof(int) * c_x->getLength());
  for(unsigned i = 0; i < c_x->getLength(); ++i){
    if (!isInf((*c_x)[i]) && i > 0) {
      allowed[i] = 1;
      ++num_allowed;
    } else { 
      allowed[i] = 0;
    }
  }

  /* determine local minimum */
  for(adj_ptr=this_->adj_list[x] ;adj_ptr != NULL; adj_ptr = adj_ptr -> succ) {
      if (!adj_ptr -> tc_valid) { 
          compute_tc_info(this_, adj_ptr);
      }

      total_impact += adj_ptr->tc_impact;

      if (num_allowed > 0) {
          for (unsigned i = 1; i < c_x->getLength(); ++i){
            if (allowed[i]){
              if (!adj_ptr->tc_safe_regs[i]){
                allowed[i] = 0;
                --num_allowed;
                if (num_allowed == 0)
                    break;
              }
            }
          }
      }
      
      if ( total_impact >= c_x->getLength() - 1 && num_allowed == 0 ) {
         result = 0;
         break;
      }
  }
  free(allowed);

  return result;
}

/* use briggs heuristic 
  note: this_ is not a general heuristic. it only is useful for 
  interference graphs.
 */
int pop_colorablenode(pbqp *this_)
{
  int deg;
  bucketnode *min_bucket=NULL;
  PBQPNum min = std::numeric_limits<PBQPNum>::infinity();
 
  /* select node where the number of colors is less than the node degree */
  for(deg=this_->max_deg;deg > 2;deg--) {
    bucketnode *bucket;
    for(bucket=this_->bucket_list[deg];bucket!= NULL;bucket = bucket -> succ) {
      int u = bucket->u;
      if (is_colorable(this_,u)) {
        pbqp_remove_bucket(this_,bucket);
        this_->num_rn_special++;
        free(bucket);
        return u; 
      } 
    }
  }

  /* select node with minimal ratio between average node costs and degree of node */
  for(deg=this_->max_deg;deg >2; deg--) {
    bucketnode *bucket;
    for(bucket=this_->bucket_list[deg];bucket!= NULL;bucket = bucket -> succ) {
      PBQPNum h;
      int u;
 
      u = bucket->u;
      assert(u>=0 && u < this_->num_nodes);
      h = (*this_->node_costs[u])[0] / (PBQPNum) deg;
      if (h < min) {
        min_bucket = bucket;
        min = h;
      } 
    }
  }

  /* return node and free bucket */
  if (min_bucket != NULL) {
    int u;

    pbqp_remove_bucket(this_,min_bucket);
    u = min_bucket->u;
    free(min_bucket);
    return u;
  } else {
    return -1;
  }
}


/*****************************************************************************
 * PBQP graph parsing
 ****************************************************************************/
 
/* reduce pbqp problem (first phase) */
static
void reduce_pbqp(pbqp *this_)
{
  int u;

  assert(this_ != NULL);
  assert(this_->bucket_list != NULL);

  for(;;){

    if (this_->bucket_list[1] != NULL) {
      u = pop_node(this_,1);
      apply_RI(this_,u); 
    } else if (this_->bucket_list[2] != NULL) {
      u = pop_node(this_,2);
      apply_RII(this_,u);
    } else if ((u = pop_colorablenode(this_)) != -1) {
      apply_RN(this_,u);
    } else {
      break;
    }
  } 
}

/*****************************************************************************
 * PBQP back propagation
 ****************************************************************************/

/* determine solution of a reduced node. Either
   RI or RII was applied for this_ node. */
static
void determine_solution(pbqp *this_,int x)
{
  PBQPVector *v = new PBQPVector(*this_ -> node_costs[x]);
  adjnode *adj_ptr;

  assert(this_ != NULL);
  assert(x >= 0 && x < this_->num_nodes);
  assert(this_ -> adj_list != NULL);
  assert(this_ -> solution != NULL);

  for(adj_ptr=this_->adj_list[x] ;adj_ptr != NULL; adj_ptr = adj_ptr -> succ) {
    int y = adj_ptr -> adj;
    int y_sol = this_ -> solution[y];

    PBQPMatrix *c_yx = pbqp_get_costmatrix(this_,y,x);
    assert(y_sol >= 0 && y_sol < (int)this_->node_costs[y]->getLength());
    (*v) += c_yx->getRowAsVector(y_sol);
    delete c_yx;
  }
  this_ -> solution[x] = v->minIndex();

  delete v;
}

/* back popagation phase of PBQP */
static
void back_propagate(pbqp *this_)
{
   int i;

   assert(this_ != NULL);
   assert(this_->stack != NULL);
   assert(this_->stack_ptr < this_->num_nodes);

   for(i=this_ -> stack_ptr-1;i>=0;i--) {
      int x = this_ -> stack[i];
      assert( x >= 0 && x < this_ -> num_nodes);
      reinsert_node(this_,x);
      determine_solution(this_,x);
   }
}

/* solve trivial nodes of degree zero */
static
void determine_trivialsolution(pbqp *this_)
{
  int u;
  PBQPNum delta;

  assert( this_ != NULL);
  assert( this_ -> bucket_list != NULL);

  /* determine trivial solution */
  while (this_->bucket_list[0] != NULL) {
    u = pop_node(this_,0);

    assert( u >= 0 && u < this_ -> num_nodes);

    this_->solution[u] = this_->node_costs[u]->minIndex();
    delta = (*this_->node_costs[u])[this_->solution[u]];
    this_->min = this_->min + delta;

    /* increment counter for number statistic */
    this_->num_r0++;
  }
}

/*****************************************************************************
 * debug facilities
 ****************************************************************************/
static
void check_pbqp(pbqp *this_)
{
  int u,v;
  PBQPMatrix *costs;
  adjnode *adj_ptr;
  
  assert( this_ != NULL);
  
  for(u=0;u< this_->num_nodes; u++) {
    assert (this_ -> node_costs[u] != NULL);
    for(adj_ptr = this_ -> adj_list[u];adj_ptr != NULL; adj_ptr = adj_ptr -> succ) {
      v = adj_ptr -> adj;
      assert( v>= 0 && v < this_->num_nodes);
      if (u < v ) {
        costs = adj_ptr -> costs;
        assert( costs->getRows() == this_->node_costs[u]->getLength() &&
                costs->getCols() == this_->node_costs[v]->getLength());
      }           
    }
  }
}

/*****************************************************************************
 * PBQP solve routines 
 ****************************************************************************/

/* solve PBQP problem */
void solve_pbqp(pbqp *this_)
{
  assert(this_ != NULL);
  assert(!this_->solved); 
  
  /* check vector & matrix dimensions */
  check_pbqp(this_);

  /* simplify PBQP problem */  
  
  /* eliminate trivial nodes, i.e.
     nodes with cost vectors of length one.  */
  eliminate_trivial_nodes(this_); 

  /* eliminate edges with independent 
     cost matrices and normalize matrices */
  eliminate_independent_edges(this_);
  
  /* create bucket list for graph parsing */
  create_bucketlist(this_);
  
  /* reduce phase */
  reduce_pbqp(this_);
  
  /* solve trivial nodes */
  determine_trivialsolution(this_);

  /* back propagation phase */
  back_propagate(this_); 
  
  this_->solved = true;
}

/* get solution of a node */
int get_pbqp_solution(pbqp *this_,int x)
{
  assert(this_ != NULL);
  assert(this_->solution != NULL);
  assert(this_ -> solved);
  
  return this_->solution[x];
}

/* is solution optimal? */
bool is_pbqp_optimal(pbqp *this_)
{
  assert(this_ -> solved);
  return this_->optimal;
}

} 

/* end of pbqp.c */
