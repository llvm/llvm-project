import StringIO


def addNodes():
    addNode("ArrayType", "Expr", "len", "Expr", "elt", "Expr")
    addNode(
        "AssignStmt",
        "Stmt",
        "lhs",
        "[]Expr",
        "rhs",
        "[]Expr",
        "define",
        "bool")
    addNode("BadDecl", "Decl")
    addNode("BadExpr", "Expr")
    addNode("BadStmt", "Stmt")
    addNode("BasicLit", "Expr", "value", "Token")
    addNode("BinaryExpr", "Expr", "x", "Expr", "y", "Expr", "op", "TokenType")
    addNode("BlockStmt", "Stmt", "list", "[]Stmt")
    addNode("Ident", "Expr", "name", "Token")
    addNode("BranchStmt", "Stmt", "label", "Ident", "tok", "TokenType")
    addNode(
        "CallExpr",
        "Expr",
        "fun",
        "Expr",
        "args",
        "[]Expr",
        "ellipsis",
        "bool")
    addNode("CaseClause", "Stmt", "list", "[]Expr", "body", "[]Stmt")
    addNode("ChanType", "Expr", "dir", "ChanDir", "value", "Expr")
    addNode("CommClause", "Stmt", "comm", "Stmt", "body", "[]Stmt")
    addNode("CompositeLit", "Expr", "type", "Expr", "elts", "[]Expr")
    addNode("DeclStmt", "Stmt", "decl", "Decl")
    addNode("DeferStmt", "Stmt", "call", "CallExpr")
    addNode("Ellipsis", "Expr", "elt", "Expr")
    addNode("EmptyStmt", "Stmt")
    addNode("ExprStmt", "Stmt", "x", "Expr")
    addNode(
        "Field",
        "Node",
        "names",
        "[]Ident",
        "type",
        "Expr",
        "tag",
        "BasicLit")
    addNode("FieldList", "Node", "list", "[]Field")
    addNode(
        "ForStmt",
        "Stmt",
        "init",
        "Stmt",
        "cond",
        "Expr",
        "post",
        "Stmt",
        "body",
        "BlockStmt")
    addNode("FuncType", "Expr", "params", "FieldList", "results", "FieldList")
    addNode(
        "FuncDecl",
        "Decl",
        "recv",
        "FieldList",
        "name",
        "Ident",
        "type",
        "FuncType",
        "body",
        "BlockStmt")
    addNode("FuncLit", "Expr", "type", "FuncType", "body", "BlockStmt")
    addNode("GenDecl", "Decl", "tok", "TokenType", "specs", "[]Spec")
    addNode("GoStmt", "Stmt", "call", "CallExpr")
    addNode(
        "IfStmt",
        "Stmt",
        "init",
        "Stmt",
        "cond",
        "Expr",
        "body",
        "BlockStmt",
        "els",
        "Stmt")
    addNode("ImportSpec", "Spec", "name", "Ident", "path", "BasicLit")
    addNode("IncDecStmt", "Stmt", "x", "Expr", "tok", "TokenType")
    addNode("IndexExpr", "Expr", "x", "Expr", "index", "Expr")
    addNode("InterfaceType", "Expr", "methods", "FieldList")
    addNode("KeyValueExpr", "Expr", "key", "Expr", "value", "Expr")
    addNode("LabeledStmt", "Stmt", "label", "Ident", "stmt", "Stmt")
    addNode("MapType", "Expr", "key", "Expr", "value", "Expr")
    addNode("ParenExpr", "Expr", "x", "Expr")
    addNode(
        "RangeStmt",
        "Stmt",
        "key",
        "Expr",
        "value",
        "Expr",
        "define",
        "bool",
        "x",
        "Expr",
        "body",
        "BlockStmt")
    addNode("ReturnStmt", "Stmt", "results", "[]Expr")
    addNode("SelectStmt", "Stmt", "body", "BlockStmt")
    addNode("SelectorExpr", "Expr", "x", "Expr", "sel", "Ident")
    addNode("SendStmt", "Stmt", "chan", "Expr", "value", "Expr")
    addNode(
        "SliceExpr",
        "Expr",
        "x",
        "Expr",
        "low",
        "Expr",
        "high",
        "Expr",
        "max",
        "Expr",
        "slice3",
        "bool")
    addNode("StarExpr", "Expr", "x", "Expr")
    addNode("StructType", "Expr", "fields", "FieldList")
    addNode(
        "SwitchStmt",
        "Stmt",
        "init",
        "Stmt",
        "tag",
        "Expr",
        "body",
        "BlockStmt")
    addNode("TypeAssertExpr", "Expr", "x", "Expr", "type", "Expr")
    addNode("TypeSpec", "Spec", "name", "Ident", "type", "Expr")
    addNode(
        "TypeSwitchStmt",
        "Stmt",
        "init",
        "Stmt",
        "assign",
        "Stmt",
        "body",
        "BlockStmt")
    addNode("UnaryExpr", "Expr", "op", "TokenType", "x", "Expr")
    addNode(
        "ValueSpec",
        "Spec",
        "names",
        "[]Ident",
        "type",
        "Expr",
        "values",
        "[]Expr")
    addParent("Decl", "Node")
    addParent("Expr", "Node")
    addParent("Spec", "Node")
    addParent("Stmt", "Node")


class Member(object):

    def __init__(self, name, typename):
        self.title = name.title()
        self.sname = name
        self.mname = 'm_' + name
        self.is_list = typename.startswith("[]")
        self.is_value = isValueType(typename)
        if self.is_value:
            self.argtype = typename
            self.mtype = typename
        elif self.is_list:
            self.argtype = 'GoAST' + typename[2:]
            self.mtype = 'std::vector<std::unique_ptr<%s> >' % self.argtype
        else:
            self.argtype = 'GoAST' + typename
            self.mtype = 'std::unique_ptr<%s>' % self.argtype
            self.mname = self.mname + '_up'


kinds = {}
parentClasses = StringIO.StringIO()
childClasses = StringIO.StringIO()
walker = StringIO.StringIO()


def startClass(name, parent, out):
    out.write("""
class GoAST%s : public GoAST%s
{
  public:
""" % (name, parent))


def endClass(name, out):
    out.write("""
    %(name)s(const %(name)s &) = delete;
    const %(name)s &operator=(const %(name)s &) = delete;
};
""" % {'name': 'GoAST' + name})


def addNode(name, parent, *children):
    startClass(name, parent, childClasses)
    l = kinds.setdefault(parent, [])
    l.append(name)
    children = createMembers(name, children)
    addConstructor(name, parent, children)
    childClasses.write("""
    const char *
    GetKindName() const override
    {
        return "%(name)s";
    }

    static bool
    classof(const GoASTNode *n)
    {
        return n->GetKind() == e%(name)s;
    }
    """ % {'name': name})
    addChildren(name, children)
    endClass(name, childClasses)


def isValueType(typename):
    if typename[0].islower():
        return True
    if typename[0].isupper():
        return typename.startswith('Token') or typename == 'ChanDir'
    return False


def createMembers(name, children):
    l = len(children)
    if (l % 2) != 0:
        raise Exception("Invalid children for %s: %s" % (name, children))
    return [Member(children[i], children[i + 1]) for i in xrange(0, l, 2)]


def addConstructor(name, parent, children):
    for c in children:
        if c.is_list:
            children = [x for x in children if x.is_value]
            break
    childClasses.write('    ')
    if len(children) == 1:
        childClasses.write('explicit ')
    childClasses.write('GoAST%s(' % name)
    for i in xrange(len(children)):
        if i > 0:
            childClasses.write(', ')

        c = children[i]
        if c.is_value:
            childClasses.write(c.argtype)
            childClasses.write(' ')
        else:
            childClasses.write('%s *' % c.argtype)
        childClasses.write(c.sname)
    childClasses.write(') : GoAST%s(e%s)' % (parent, name))
    for c in children:
        childClasses.write(', ')
        childClasses.write('%(mname)s(%(sname)s)' % c.__dict__)
    childClasses.write(""" {}
    ~GoAST%s() override = default;
""" % name)


def addChildren(name, children):
    if len(children) == 0:
        return
    walker.write("""
    case e%(n)s:
        {
            GoAST%(n)s *n = llvm::cast<GoAST%(n)s>(this);
            (void)n;""" % {'n': name})
    for c in children:
        if c.is_list:
            childClasses.write("""
    size_t
    Num%(title)s() const
    {
        return %(mname)s.size();
    }
    const %(argtype)s *
    Get%(title)s(int i) const
    {
        return %(mname)s[i].get();
    }
    void
    Add%(title)s(%(argtype)s *%(sname)s)
    {
        %(mname)s.push_back(std::unique_ptr<%(argtype)s>(%(sname)s));
    }
""" % c.__dict__)
            walker.write("""
            for (auto& e : n->%s) { v(e.get()); }""" % c.mname)
        else:
            const = ''
            get = ''
            set = ''
            t = c.argtype
            if isValueType(t):
                set = '%(mname)s = %(sname)s' % c.__dict__
                t = t + ' '
            else:
                t = t + ' *'
                const = 'const '
                get = '.get()'
                set = '%(mname)s.reset(%(sname)s)' % c.__dict__
                walker.write("""
            v(n->%s.get());""" % c.mname)
            childClasses.write("""
    %(const)s%(type)s
    Get%(title)s() const
    {
        return %(mname)s%(get)s;
    }
    void
    Set%(title)s(%(type)s%(sname)s)
    {
        %(set)s;
    }
""" % {'const': const, 'title': c.title, 'sname': c.sname, 'get': get, 'set': set, 'type': t, 'mname': c.mname})
    childClasses.write('\n  private:\n    friend class GoASTNode;\n')
    walker.write("""
            return;
        }""")
    for c in children:
        childClasses.write('    %s %s;\n' % (c.mtype, c.mname))


def addParent(name, parent):
    startClass(name, parent, parentClasses)
    l = kinds[name]
    minName = l[0]
    maxName = l[-1]
    parentClasses.write("""    template <typename R, typename V> R Visit(V *v) const;

    static bool
    classof(const GoASTNode *n)
    {
        return n->GetKind() >= e%s && n->GetKind() <= e%s;
    }

  protected:
    explicit GoAST%s(NodeKind kind) : GoASTNode(kind) { }
  private:
""" % (minName, maxName, name))
    endClass(name, parentClasses)

addNodes()

print """//===-- GoAST.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// DO NOT EDIT.
// Generated by gen_go_ast.py

#ifndef liblldb_GoAST_h
#define liblldb_GoAST_h

#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"
#include "llvm/Support/Casting.h"
#include "Plugins/ExpressionParser/Go/GoLexer.h"

namespace lldb_private
{

class GoASTNode
{
  public:
    typedef GoLexer::TokenType TokenType;
    typedef GoLexer::Token Token;
    enum ChanDir
    {
        eChanBidir,
        eChanSend,
        eChanRecv,
    };
    enum NodeKind
    {"""
for l in kinds.itervalues():
    for x in l:
        print "        e%s," % x
print """    };

    virtual ~GoASTNode() = default;

    NodeKind
    GetKind() const
    {
        return m_kind;
    }

    virtual const char *GetKindName() const = 0;

    template <typename V> void WalkChildren(V &v);

  protected:
    explicit GoASTNode(NodeKind kind) : m_kind(kind) { }

  private:
    const NodeKind m_kind;

    GoASTNode(const GoASTNode &) = delete;
    const GoASTNode &operator=(const GoASTNode &) = delete;
};
"""


print parentClasses.getvalue()
print childClasses.getvalue()

for k, l in kinds.iteritems():
    if k == 'Node':
        continue
    print """
template <typename R, typename V>
R GoAST%s::Visit(V* v) const
{
    switch(GetKind())
    {""" % k
    for subtype in l:
        print """    case e%(n)s:
        return v->Visit%(n)s(llvm::cast<const GoAST%(n)s>(this));""" % {'n': subtype}

    print """    default:
        assert(false && "Invalid kind");
    }
}"""

print """
template <typename V>
void GoASTNode::WalkChildren(V &v)
{
    switch (m_kind)
    {
"""
print walker.getvalue()
print"""
        case eEmptyStmt:
        case eBadDecl:
        case eBadExpr:
        case eBadStmt:
          break;
    }
}

}  // namespace lldb_private

#endif
"""
