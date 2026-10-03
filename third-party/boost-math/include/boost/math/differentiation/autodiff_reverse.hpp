//           Copyright Maksym Zhelyeznyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_MATH_DIFFERENTIATION_AUTODIFF_REVERSE_HPP
#define BOOST_MATH_DIFFERENTIATION_AUTODIFF_REVERSE_HPP

#include <boost/math/constants/constants.hpp>

#if defined(BOOST_MATH_REVERSE_MODE_ET_OFF) && defined(BOOST_MATH_REVERSE_MODE_ET_ON)
#error "Cannot define both BOOST_MATH_REVERSE_MODE_ET_OFF and BOOST_MATH_REVERSE_MODE_ET_ON"
#endif

#if !defined(BOOST_MATH_REVERSE_MODE_ET_OFF) && !defined(BOOST_MATH_REVERSE_MODE_ET_ON)
#define BOOST_MATH_REVERSE_MODE_ET_ON
#endif

#ifdef BOOST_MATH_REVERSE_MODE_ET_ON
#include <boost/math/differentiation/detail/reverse_mode_autodiff_basic_ops_et.hpp>
#include <boost/math/differentiation/detail/reverse_mode_autodiff_stl_et.hpp>
#else
#include <boost/math/differentiation/detail/reverse_mode_autodiff_basic_ops_no_et.hpp>
#include <boost/math/differentiation/detail/reverse_mode_autodiff_stl_no_et.hpp>
#endif

#include <boost/math/differentiation/detail/reverse_mode_autodiff_comparison_operator_overloads.hpp>
#include <boost/math/differentiation/detail/reverse_mode_autodiff_erf_overloads.hpp>
#include <boost/math/differentiation/detail/reverse_mode_autodiff_expression_template_base.hpp>
#include <boost/math/differentiation/detail/reverse_mode_autodiff_memory_management.hpp>
#include <boost/math/special_functions/acosh.hpp>
#include <boost/math/special_functions/asinh.hpp>
#include <boost/math/special_functions/atanh.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/math/special_functions/trunc.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/math/tools/promotion.hpp>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <vector>
#define BOOST_MATH_BUFFER_SIZE 65536

namespace boost {
namespace math {
namespace differentiation {
namespace reverse_mode {

/* forward declarations for utitlity functions */
template<typename RealType, size_t DerivativeOrder, class DerivedExpression>
struct expression;

template<typename RealType, size_t DerivativeOrder>
class rvar;

template<typename RealType,
         size_t DerivativeOrder,
         typename LHS,
         typename RHS,
         typename ConcreteBinaryOperation>
struct abstract_binary_expression;

template<typename RealType, size_t DerivativeOrder, typename ARG, typename ConcreteBinaryOperation>
struct abstract_unary_expression;

template<typename RealType, size_t DerivativeOrder>
class gradient_node; // forward declaration for tape
// manages nodes in computational graph
template<typename RealType, size_t DerivativeOrder, size_t buffer_size = BOOST_MATH_BUFFER_SIZE>
class gradient_tape
{
    /** @brief tape (graph) management class for autodiff
   *  holds all the data structures for autodiff */
private:
    /* type decays to order - 1 to support higher order derivatives */
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;
    /* adjoints are the overall derivative, and derivatives are the "local"
   * derivative */
    detail::flat_linear_allocator<inner_t, buffer_size>                   adjoints_;
    detail::flat_linear_allocator<inner_t, buffer_size>                   derivatives_;
    detail::flat_linear_allocator<gradient_node<RealType, DerivativeOrder>, buffer_size>
        gradient_nodes_;
    detail::flat_linear_allocator<gradient_node<RealType, DerivativeOrder> *, buffer_size>
        argument_nodes_;

    // compile time check if emplace_back calls on zero
    template<size_t n>
    gradient_node<RealType, DerivativeOrder> *fill_node_at_compile_time(
        std::true_type, gradient_node<RealType, DerivativeOrder> *node_ptr)
    {
        node_ptr->derivatives_    = derivatives_.template emplace_back_n<n>();
        node_ptr->argument_nodes_ = argument_nodes_.template emplace_back_n<n>();
        return node_ptr;
    }

    template<size_t n>
    gradient_node<RealType, DerivativeOrder> *fill_node_at_compile_time(
        std::false_type, gradient_node<RealType, DerivativeOrder> *node_ptr)
    {
        node_ptr->derivatives_       = nullptr;
        node_ptr->argument_adjoints_ = nullptr;
        node_ptr->argument_nodes_    = nullptr;
        return node_ptr;
    }

public:
    /* gradient node stores iterators to its data memebers
   * (adjoint/derivative/arguments) so that in case flat linear allocator
   * reaches its block boundary and needs more memory for that node, the
   * iterator can be invoked to access it */
    using adjoint_iterator = typename detail::flat_linear_allocator<inner_t, buffer_size>::iterator;
    using derivatives_iterator =
        typename detail::flat_linear_allocator<inner_t, buffer_size>::iterator;
    using gradient_nodes_iterator =
        typename detail::flat_linear_allocator<gradient_node<RealType, DerivativeOrder>,
                                               buffer_size>::iterator;
    using argument_nodes_iterator =
        typename detail::flat_linear_allocator<gradient_node<RealType, DerivativeOrder> *,
                                               buffer_size>::iterator;

    gradient_tape() { clear(); };

    gradient_tape(const gradient_tape &)            = delete;
    gradient_tape &operator=(const gradient_tape &) = delete;
    gradient_tape(gradient_tape &&other)            = delete;
    gradient_tape operator=(gradient_tape &&other)  = delete;
    ~gradient_tape() noexcept { clear(); }
    void clear()
    {
        adjoints_.clear();
        derivatives_.clear();
        gradient_nodes_.clear();
        argument_nodes_.clear();
    }

    // no derivatives or arguments
    gradient_node<RealType, DerivativeOrder> *emplace_leaf_node()
    {
        gradient_node<RealType, DerivativeOrder> *node = &*gradient_nodes_.emplace_back();
        node->adjoint_                = adjoints_.emplace_back();
        node->derivatives_            = derivatives_iterator();    // nullptr;
        node->argument_nodes_         = argument_nodes_iterator(); // nullptr;

        return node;
    };

    // single argument, single derivative
    gradient_node<RealType, DerivativeOrder> *emplace_active_unary_node()
    {
        gradient_node<RealType, DerivativeOrder> *node = &*gradient_nodes_.emplace_back();
        node->n_                      = 1;
        node->adjoint_                = adjoints_.emplace_back();
        node->derivatives_            = derivatives_.emplace_back();

        return node;
    };

    // arbitrary number of arguments/derivatives (compile time)
    template<size_t n>
    gradient_node<RealType, DerivativeOrder> *emplace_active_multi_node()
    {
        gradient_node<RealType, DerivativeOrder> *node = &*gradient_nodes_.emplace_back();
        node->n_                      = n;
        node->adjoint_                = adjoints_.emplace_back();
        // emulate if constexpr
        return fill_node_at_compile_time<n>(std::integral_constant<bool, (n > 0)>{}, node);
    }

    // same as above at runtime
    gradient_node<RealType, DerivativeOrder> *emplace_active_multi_node(size_t n)
    {
        gradient_node<RealType, DerivativeOrder> *node = &*gradient_nodes_.emplace_back();
        node->n_                      = n;
        node->adjoint_                = adjoints_.emplace_back();
        if (n > 0) {
            node->derivatives_    = derivatives_.emplace_back_n(n);
            node->argument_nodes_ = argument_nodes_.emplace_back_n(n);
        }
        return node;
    };
    /* manual reset button for all adjoints */
    void zero_grad()
    {
        const RealType zero = RealType(0.0);
        adjoints_.fill(zero);
    }

    // return type is an iterator
    auto begin() { return gradient_nodes_.begin(); }
    auto end() { return gradient_nodes_.end(); }
    auto find(gradient_node<RealType, DerivativeOrder> *node)
    {
        return gradient_nodes_.find(node);
    };
    void add_checkpoint()
    {
        gradient_nodes_.add_checkpoint();
        adjoints_.add_checkpoint();
        derivatives_.add_checkpoint();
        argument_nodes_.add_checkpoint();
    };

    auto last_checkpoint() { return gradient_nodes_.last_checkpoint(); };
    auto first_checkpoint() { return gradient_nodes_.last_checkpoint(); };
    auto checkpoint_at(size_t index) { return gradient_nodes_.get_checkpoint_at(index); };
    void rewind_to_last_checkpoint()
    {
        gradient_nodes_.rewind_to_last_checkpoint();
        adjoints_.rewind_to_last_checkpoint();
        derivatives_.rewind_to_last_checkpoint();
        argument_nodes_.rewind_to_last_checkpoint();
    };
    void rewind_to_checkpoint_at(size_t index) // index is "checkpoint" index. so
                                               // order which checkpoint was set
    {
        gradient_nodes_.rewind_to_checkpoint_at(index);
        adjoints_.rewind_to_checkpoint_at(index);
        derivatives_.rewind_to_checkpoint_at(index);
        argument_nodes_.rewind_to_checkpoint_at(index);
    }

    // rewind to beginning of computational graph
    void rewind()
    {
        gradient_nodes_.rewind();
        adjoints_.rewind();
        derivatives_.rewind();
        argument_nodes_.rewind();
    }

    // random acces
    gradient_node<RealType, DerivativeOrder> &operator[](size_t i) { return gradient_nodes_[i]; }
    const gradient_node<RealType, DerivativeOrder> &operator[](size_t i) const
    {
        return gradient_nodes_[i];
    }
};
// class rvar;
template<typename RealType, size_t DerivativeOrder> // no CRTP, just storage
class gradient_node
{
    /*
   * @brief manages adjoints, derivatives, and stores points to argument
   * adjoints pointers to arguments aren't needed here
   * */
public:
    using adjoint_iterator = typename gradient_tape<RealType, DerivativeOrder>::adjoint_iterator;
    using derivatives_iterator =
        typename gradient_tape<RealType, DerivativeOrder>::derivatives_iterator;
    using argument_nodes_iterator =
        typename gradient_tape<RealType, DerivativeOrder>::argument_nodes_iterator;

private:
    size_t n_;
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;
    /* these are iterators in case
   * flat linear allocator is at capacity, and needs to allocate a new block of
   * memory. */
    adjoint_iterator        adjoint_;
    derivatives_iterator    derivatives_;
    argument_nodes_iterator argument_nodes_;

public:
    friend class gradient_tape<RealType, DerivativeOrder>;
    friend class rvar<RealType, DerivativeOrder>;

    gradient_node() = default;
    explicit gradient_node(const size_t n)
        : n_(n)
        , adjoint_(nullptr)
        , derivatives_(nullptr)
    {}
    explicit gradient_node(const size_t                      n,
                           RealType                         *adjoint,
                           RealType                         *derivatives,
                           rvar<RealType, DerivativeOrder> **arguments)
        : n_(n)
        , adjoint_(adjoint)
        , derivatives_(derivatives)
    { static_cast<void>(arguments); }

    inner_t get_adjoint_v() const { return *adjoint_; }
    inner_t get_derivative_v(size_t arg_id) const { return derivatives_[static_cast<ptrdiff_t>(arg_id)]; };
    inner_t get_argument_adjoint_v(size_t arg_id) const
    {
        return *argument_nodes_[static_cast<ptrdiff_t>(arg_id)]->adjoint_;
    }

    adjoint_iterator get_adjoint_ptr() { return adjoint_; }
    adjoint_iterator get_adjoint_ptr() const { return adjoint_; };
    void             update_adjoint_v(inner_t value) { *adjoint_ = value; };
    void update_derivative_v(size_t arg_id, inner_t value) { derivatives_[static_cast<ptrdiff_t>(arg_id)] = value; };
    void update_argument_adj_v(size_t arg_id, inner_t value)
    {
        argument_nodes_[static_cast<ptrdiff_t>(arg_id)]->update_adjoint_v(value);
    };
    void update_argument_ptr_at(size_t arg_id, gradient_node<RealType, DerivativeOrder> *node_ptr)
    {
        argument_nodes_[static_cast<ptrdiff_t>(arg_id)] = node_ptr;
    }

    void backward()
    {
        if (!n_) // leaf node
            return;

        using boost::math::differentiation::reverse_mode::fabs;
        using std::fabs;
        if (!adjoint_ || fabs(*adjoint_) < 2 * std::numeric_limits<RealType>::epsilon())
            return;

        if (!argument_nodes_) // no arguments
            return;

        if (!derivatives_) // no derivatives
            return;

        for (size_t i = 0; i < n_; ++i) {
            auto adjoint          = get_adjoint_v();
            auto derivative       = get_derivative_v(i);
            auto argument_adjoint = get_argument_adjoint_v(i);
            update_argument_adj_v(i, argument_adjoint + derivative * adjoint);
        }
    }
};

/****************************************************************************************************************/
template<typename RealType, size_t DerivativeOrder>
inline gradient_tape<RealType, DerivativeOrder, BOOST_MATH_BUFFER_SIZE> &get_active_tape()
{
    static BOOST_MATH_THREAD_LOCAL gradient_tape<RealType, DerivativeOrder, BOOST_MATH_BUFFER_SIZE>
                                   tape;
    return tape;
}

template<typename RealType, size_t DerivativeOrder = 1>
class rvar : public expression<RealType, DerivativeOrder, rvar<RealType, DerivativeOrder>>
{
private:
    using inner_t = rvar_t<RealType, DerivativeOrder - 1>;
    friend class gradient_node<RealType, DerivativeOrder>;
    inner_t                  value_;
    gradient_node<RealType, DerivativeOrder> *node_ = nullptr;
    template<typename, size_t>
    friend class rvar;
    /*****************************************************************************************/
    /**
     * @brief implementation helpers for get_value_at
     */
    template<size_t target_order, size_t current_order>
    struct get_value_at_impl
    {
        static_assert(target_order <= current_order, "Requested depth exceeds variable order.");

        /** @return value_ at rvar_t<T,current_order - 1>
         */
        static auto &get(rvar<RealType, current_order> &v)
        {
            return get_value_at_impl<target_order, current_order - 1>::get(v.get_value());
        }
        /** @return const value_ at rvar_t<T,current_order - 1>
         */
        static const auto &get(const rvar<RealType, current_order> &v)
        {
            return get_value_at_impl<target_order, current_order - 1>::get(v.get_value());
        }
    };

    /** @brief base case specialization for target_order == current order
     */
    template<size_t target_order>
    struct get_value_at_impl<target_order, target_order>
    {
        /** @return value_ at rvar_t<T,target_order>
         */
        static auto       &get(rvar<RealType, target_order> &v) { return v; }
        /** @return const value_ at rvar_t<T,target_order>
         */
        static const auto &get(const rvar<RealType, target_order> &v) { return v; }
    };
    /*****************************************************************************************/
    void make_leaf_node()
    {
        gradient_tape<RealType, DerivativeOrder, BOOST_MATH_BUFFER_SIZE> &tape
            = get_active_tape<RealType, DerivativeOrder>();
        node_                                      = tape.emplace_leaf_node();
    }

    void make_unary_node()
    {
        gradient_tape<RealType, DerivativeOrder, BOOST_MATH_BUFFER_SIZE> &tape
            = get_active_tape<RealType, DerivativeOrder>();
        node_                                      = tape.emplace_active_unary_node();
    }

    void make_multi_node(size_t n)
    {
        gradient_tape<RealType, DerivativeOrder, BOOST_MATH_BUFFER_SIZE> &tape
            = get_active_tape<RealType, DerivativeOrder>();
        node_                                      = tape.emplace_active_multi_node(n);
    }

    template<size_t n>
    void make_multi_node()
    {
        gradient_tape<RealType, DerivativeOrder, BOOST_MATH_BUFFER_SIZE> &tape
            = get_active_tape<RealType, DerivativeOrder>();
        node_                                      = tape.template emplace_active_multi_node<n>();
    }

    template<typename E>
    void make_rvar_from_expr(const expression<RealType, DerivativeOrder, E> &expr)
    {
        make_multi_node<detail::count_rvars<E, DerivativeOrder>>();
        expr.template propagatex<0>(node_, inner_t(static_cast<RealType>(1.0)));
    }
    RealType get_item_impl(std::true_type) const
    {
        return value_.get_item_impl(std::integral_constant<bool, (DerivativeOrder - 1 > 1)>{});
    }

    RealType get_item_impl(std::false_type) const { return value_; }

public:
    using value_type                          = RealType;
    static constexpr size_t DerivativeOrder_v = DerivativeOrder;
    rvar()
        : value_()
    {
        make_leaf_node();
    }
    rvar(const RealType value)
        : value_(inner_t{static_cast<RealType>(value)})
    {
        make_leaf_node();
    }

    rvar &operator=(RealType v)
    {
        value_ = inner_t(v);
        if (node_ == nullptr) {
            make_leaf_node();
        }
        return *this;
    }
    rvar(const rvar<RealType, DerivativeOrder> &other)            = default;
    rvar &operator=(const rvar<RealType, DerivativeOrder> &other) = default;

    template<size_t arg_index>
    void propagatex(gradient_node<RealType, DerivativeOrder> *node, inner_t adj) const
    {
        node->update_derivative_v(arg_index, adj);
        node->update_argument_ptr_at(arg_index, node_);
    }

    template<class E>
    rvar(const expression<RealType, DerivativeOrder, E> &expr)
    {
        value_ = expr.evaluate();
        make_rvar_from_expr(expr);
    }

    template<typename T,
             typename = std::enable_if_t<is_floating_point_v<T> && !is_same_v<T, RealType>>>
    rvar(T v)
        : value_(inner_t{static_cast<RealType>(v)})
    {
        make_leaf_node();
    }

    template<class E>
    rvar &operator=(const expression<RealType, DerivativeOrder, E> &expr)
    {
        value_ = expr.evaluate();
        make_rvar_from_expr(expr);
        return *this;
    }
    /***************************************************************************************************/
    template<class E>
    rvar<RealType, DerivativeOrder> &operator+=(const expression<RealType, DerivativeOrder, E> &expr)
    {
        *this = *this + expr;
        return *this;
    }

    template<class E>
    rvar<RealType, DerivativeOrder> &operator*=(const expression<RealType, DerivativeOrder, E> &expr)
    {
        *this = *this * expr;
        return *this;
    }

    template<class E>
    rvar<RealType, DerivativeOrder> &operator-=(const expression<RealType, DerivativeOrder, E> &expr)
    {
        *this = *this - expr;
        return *this;
    }

    template<class E>
    rvar<RealType, DerivativeOrder> &operator/=(const expression<RealType, DerivativeOrder, E> &expr)
    {
        *this = *this / expr;
        return *this;
    }
    /***************************************************************************************************/
    rvar<RealType, DerivativeOrder> &operator+=(const RealType &v)
    {
        *this = *this + v;
        return *this;
    }

    rvar<RealType, DerivativeOrder> &operator*=(const RealType &v)
    {
        *this = *this * v;
        return *this;
    }

    rvar<RealType, DerivativeOrder> &operator-=(const RealType &v)
    {
        *this = *this - v;
        return *this;
    }

    rvar<RealType, DerivativeOrder> &operator/=(const RealType &v)
    {
        *this = *this / v;
        return *this;
    }

    /***************************************************************************************************/
    const inner_t &adjoint() const { return *node_->get_adjoint_ptr(); }
    inner_t       &adjoint() { return *node_->get_adjoint_ptr(); }

    const inner_t &evaluate() const { return value_; };
    inner_t       &get_value() { return value_; };

    explicit operator RealType() const { return item(); }

    explicit       operator int() const { return static_cast<int>(item()); }
    explicit       operator long() const { return static_cast<long>(item()); }
    explicit       operator long long() const { return static_cast<long long>(item()); }

    /**
     *  @brief same as evaluate but returns proper depth for higher order derivatives
     *  @return value_ at depth N
     */
    template<size_t N>
    auto &get_value_at()
    {
        static_assert(N <= DerivativeOrder, "Requested depth exceeds variable order.");
        return get_value_at_impl<N, DerivativeOrder>::get(*this);
    }
    /** @brief same as above but const
     */
    template<size_t N>
    const auto &get_value_at() const
    {
        static_assert(N <= DerivativeOrder, "Requested depth exceeds variable order.");
        return get_value_at_impl<N, DerivativeOrder>::get(*this);
    }

    RealType item() const
    {
        return get_item_impl(std::integral_constant<bool, (DerivativeOrder > 1)>{});
    }

    void backward()
    {
        gradient_tape<RealType, DerivativeOrder, BOOST_MATH_BUFFER_SIZE> &tape
            = get_active_tape<RealType, DerivativeOrder>();
        auto                                  it   = tape.find(node_);
        it->update_adjoint_v(inner_t(static_cast<RealType>(1.0)));
        while (it != tape.begin()) {
            it->backward();
            --it;
        }
        it->backward();
    }

    void set_item(RealType value) { value_ = inner_t(value); }
};

template<typename RealType, size_t DerivativeOrder>
std::ostream &operator<<(std::ostream &os, const rvar<RealType, DerivativeOrder> var)
{
    os << "rvar<" << DerivativeOrder << ">(" << var.item() << "," << var.adjoint() << ")";
    return os;
}

template<typename RealType, size_t DerivativeOrder, typename E>
std::ostream &operator<<(std::ostream &os, const expression<RealType, DerivativeOrder, E> &expr)
{
    rvar<RealType, DerivativeOrder> tmp = expr;
    os << "rvar<" << DerivativeOrder << ">(" << tmp.item() << "," << tmp.adjoint() << ")";
    return os;
}

template<typename RealType, size_t DerivativeOrder>
rvar<RealType, DerivativeOrder> make_rvar(const RealType v)
{
    static_assert(DerivativeOrder > 0, "rvar order must be >= 1");
    return rvar<RealType, DerivativeOrder>(v);
}
template<typename RealType, size_t DerivativeOrder, typename E>
rvar<RealType, DerivativeOrder> make_rvar(const expression<RealType, DerivativeOrder, E> &expr)
{
    static_assert(DerivativeOrder > 0, "rvar order must be >= 1");
    return rvar<RealType, DerivativeOrder>(expr);
}

namespace detail {

/** @brief helper overload for grad implementation.
 *  @return vector<rvar<T,order-1> of gradients of the autodiff graph.
 *  specialization for autodiffing through autodiff. i.e. being able to
 *  compute higher order grads
*/
template<typename RealType, size_t DerivativeOrder>
struct grad_op_impl
{
    std::vector<rvar<RealType, DerivativeOrder - 1>> operator()(
        rvar<RealType, DerivativeOrder> &f, std::vector<rvar<RealType, DerivativeOrder> *> &x)
    {
        auto &tape = get_active_tape<RealType, DerivativeOrder>();
        tape.zero_grad();
        f.backward();

        std::vector<rvar<RealType, DerivativeOrder - 1>> gradient_vector;
        gradient_vector.reserve(x.size());

        for (auto &xi : x) {
            gradient_vector.emplace_back(xi->adjoint());
        }
        return gradient_vector;
    }
};
/** @brief helper overload for grad implementation.
 *  @return vector<T> of gradients of the autodiff graph.
 *          base specialization for order 1 autodiff
*/
template<typename T>
struct grad_op_impl<T, 1>
{
    std::vector<T> operator()(rvar<T, 1> &f, std::vector<rvar<T, 1> *> &x)
    {
        gradient_tape<T, 1, BOOST_MATH_BUFFER_SIZE> &tape = get_active_tape<T, 1>();
        tape.zero_grad();
        f.backward();
        std::vector<T> gradient_vector;
        gradient_vector.reserve(x.size());
        for (auto &xi : x) {
            gradient_vector.push_back(xi->adjoint());
        }
        return gradient_vector;
    }
};

/** @brief helper overload for higher order autodiff
 *  @return nested vector representing N-d tensor of
 *      higher order derivatives
 */
template<size_t N,
         typename RealType,
         size_t DerivativeOrder_1,
         size_t DerivativeOrder_2,
         typename Enable = void>
struct grad_nd_impl
{
    auto operator()(rvar<RealType, DerivativeOrder_1>                &f,
                    std::vector<rvar<RealType, DerivativeOrder_2> *> &x)
    {
        static_assert(N > 1, "N must be greater than 1 for this template");

        auto current_grad = grad(f, x); // vector<rvar<T,DerivativeOrder_1-1>> or vector<T>

        std::vector<decltype(grad_nd_impl<N - 1, RealType, DerivativeOrder_1 - 1, DerivativeOrder_2>()(
            current_grad[0], x))>
            result;
        result.reserve(current_grad.size());

        for (auto &g_i : current_grad) {
            result.push_back(
                grad_nd_impl<N - 1, RealType, DerivativeOrder_1 - 1, DerivativeOrder_2>()(g_i, x));
        }
        return result;
    }
};
/** @brief spcialization for order = 1,
 *  @return vector<rvar<T,DerivativeOrder_1-1>> gradients */
template<typename RealType, size_t DerivativeOrder_1, size_t DerivativeOrder_2>
struct grad_nd_impl<1, RealType, DerivativeOrder_1, DerivativeOrder_2>
{
    auto operator()(rvar<RealType, DerivativeOrder_1>                &f,
                    std::vector<rvar<RealType, DerivativeOrder_2> *> &x)
    {
        return grad(f, x);
    }
};

template<typename ptr>
struct rvar_order;

template<typename RealType, size_t DerivativeOrder>
struct rvar_order<rvar<RealType, DerivativeOrder> *>
{
    static constexpr size_t value = DerivativeOrder;
};

} // namespace detail

/**
 * @brief grad computes gradient with respect to vector of pointers x
 * @param f -> computational graph
 * @param x -> variables gradients to record. Note ALL gradients of the graph
 *             are computed simultaneously, only the ones w.r.t. x are returned
 * @return vector<rvar<T,DerivativeOrder_1 - 1> of gradients. in the case of DerivativeOrder_1 = 1
 *            rvar<T,DerivativeOrder_1-1> decays to T
 *
 * safe to call recursively with grad(grad(grad...
 */
template<typename RealType, size_t DerivativeOrder_1, size_t DerivativeOrder_2>
auto grad(rvar<RealType, DerivativeOrder_1> &f, std::vector<rvar<RealType, DerivativeOrder_2> *> &x)
{
    static_assert(DerivativeOrder_1 <= DerivativeOrder_2,
                  "variable differentiating w.r.t. must have order >= function order");
    std::vector<rvar<RealType, DerivativeOrder_1> *> xx;
    xx.reserve(x.size());
    for (auto &xi : x)
        xx.push_back(&(xi->template get_value_at<DerivativeOrder_1>()));
    return detail::grad_op_impl<RealType, DerivativeOrder_1>{}(f, xx);
}
/** @brief variadic overload of above
 */
template<typename RealType, size_t DerivativeOrder_1, typename First, typename... Other>
auto grad(rvar<RealType, DerivativeOrder_1> &f, First first, Other... other)
{
    constexpr size_t DerivativeOrder_2 = detail::rvar_order<First>::value;
    static_assert(DerivativeOrder_1 <= DerivativeOrder_2,
                  "variable differentiating w.r.t. must have order >= function order");
    std::vector<rvar<RealType, DerivativeOrder_2> *> x_vec = {first, other...};
    return grad(f, x_vec);
}

/** @brief computes hessian matrix of computational graph w.r.t.
 *         vector of variables x.
 *  @return std::vector<std::vector<rvar<T,DerivativeOrder_1-2>> hessian matrix
 *          rvar<T,2> decays to T
 *
 *  NOT recursion safe, cannot do hess(hess(
 */
template<typename RealType, size_t DerivativeOrder_1, size_t DerivativeOrder_2>
auto hess(rvar<RealType, DerivativeOrder_1> &f, std::vector<rvar<RealType, DerivativeOrder_2> *> &x)
{
    return detail::grad_nd_impl<2, RealType, DerivativeOrder_1, DerivativeOrder_2>{}(f, x);
}
/** @brief variadic overload of above
 */
template<typename RealType, size_t DerivativeOrder_1, typename First, typename... Other>
auto hess(rvar<RealType, DerivativeOrder_1> &f, First first, Other... other)
{
    constexpr size_t DerivativeOrder_2                     = detail::rvar_order<First>::value;
    std::vector<rvar<RealType, DerivativeOrder_2> *> x_vec = {first, other...};
    return hess(f, x_vec);
}

/** @brief comput N'th gradient of computational graph w.r.t. x
 *  @return vector<vector<.... up N nestings representing tensor
 *          of gradients of order N
 *
 *  NOT recursively safe, cannot do grad_nd(grad_nd(... etc...
 */
template<size_t N, typename RealType, size_t DerivativeOrder_1, size_t DerivativeOrder_2>
auto grad_nd(rvar<RealType, DerivativeOrder_1>                &f,
             std::vector<rvar<RealType, DerivativeOrder_2> *> &x)
{
    static_assert(DerivativeOrder_1 >= N, "Function order must be at least N");
    static_assert(DerivativeOrder_2 >= DerivativeOrder_1,
                  "Variable order must be at least function order");

    return detail::grad_nd_impl<N, RealType, DerivativeOrder_1, DerivativeOrder_2>()(f, x);
}

/** @brief variadic overload of above
 */
template<size_t N, typename ftype, typename First, typename... Other>
auto grad_nd(ftype &f, First first, Other... other)
{
    using RealType                                         = typename ftype::value_type;
    constexpr size_t DerivativeOrder_1                     = detail::rvar_order<ftype *>::value;
    constexpr size_t DerivativeOrder_2                     = detail::rvar_order<First>::value;
    std::vector<rvar<RealType, DerivativeOrder_2> *> x_vec = {first, other...};
    return detail::grad_nd_impl<N, RealType, DerivativeOrder_1, DerivativeOrder_1>{}(f, x_vec);
}
} // namespace reverse_mode
} // namespace differentiation
} // namespace math
} // namespace boost
namespace std {

// copied from forward mode
template<typename RealType, size_t DerivativeOrder>
class numeric_limits<boost::math::differentiation::reverse_mode::rvar<RealType, DerivativeOrder>>
    : public numeric_limits<typename boost::math::differentiation::reverse_mode::
                                rvar<RealType, DerivativeOrder>::value_type>
{};
} // namespace std
#endif
