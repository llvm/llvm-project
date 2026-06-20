/* ============================================================
   LLVM Advisor — Shell (Topbar, Sidebar, Detail Panel)
   ============================================================ */

const Shell = {
  init() {
    const app = document.getElementById('app');
    clearEl(app);
    Theme.init();
    app.appendChild(this.topbar());
    app.appendChild(this.sidebar());
    const wrap = h('div', { id: 'main-wrap' },
      h('div', { id: 'main' }),
      h('div', { id: 'detail-panel' })
    );
    app.appendChild(wrap);
    app.appendChild(CommandPalette.render());

    // React to state
    State.on('sidebarPinned', v => app.classList.toggle('sb-exp', v));
    State.on('detailOpen', v => app.classList.toggle('detail-open', v));
    State.on('commandPaletteOpen', v => {
      document.querySelector('.cmd-overlay')?.classList.toggle('open', v);
      if (v) document.querySelector('.cmd-input')?.focus();
    });
    State.on('route', () => this.updateNav());
    State.on('health', h => this.updateStatus(h));

    // Load initial data
    this.loadData();
  },

  async loadData() {
    const [health, snaps] = await Promise.all([API.health(), API.snapshots()]);
    if (health.ok) State.set('health', health.data);
    if (snaps.ok) {
      const list = Array.isArray(snaps.data) ? snaps.data : [];
      list.sort((a, b) => (b.created_unix || 0) - (a.created_unix || 0));
      State.set('snapshots', list);
      if (list.length > 0 && !State.get('currentSnapshot')) {
        State.set('currentSnapshot', list[0]);
        this.updateSelectors();
        Router.resolve();
        return;
      }
    }
    this.updateSelectors();
  },

  topbar() {
    return h('header', { id: 'topbar' },
      h('div', { class: 'wordmark', onClick: () => Router.navigate('/') }, 'llvm-advisor'),
      h('div', { class: 'topbar-center' },
        h('div', { class: 'dropdown', id: 'snapshot-dropdown' },
          h('button', { class: 'dd-trigger', onClick: e => this.toggleDropdown(e) }, 'Snapshot ▾'),
          h('div', { class: 'dd-menu' })
        )
      ),
      h('div', { class: 'topbar-right' },
        h('div', { class: 'status-pill', id: 'status-pill' },
          h('span', { class: 'status-dot ok' }),
          h('span', {}, 'connecting…')
        ),
        h('button', { class: 'theme-toggle', id: 'theme-toggle', title: 'Toggle theme', onClick: () => { Theme.toggle(); this.updateThemeIcon(); } },
          renderIcon(Icons[Theme.icon()])
        ),
        h('button', { class: 'mono', style: { fontSize: '12px', color: 'var(--text-muted)' },
          onClick: () => State.set('commandPaletteOpen', true) }, '?')
      )
    );
  },

  sidebar() {
    const items = [
      { icon: 'overview', label: 'Overview', route: '/', shortcut: 'g o' },
      { icon: 'units', label: 'Units', route: '/units', shortcut: 'g u' },
      { icon: 'compare', label: 'Compare', route: '/compare', shortcut: 'g c' },
      { icon: 'timeline', label: 'Timeline', route: '/timeline', shortcut: 'g t' },
      { icon: 'insights', label: 'Insights', route: '/insights', shortcut: 'g i' },
      { icon: 'settings', label: 'Settings', route: '/settings', shortcut: 'g s' },
    ];

    const nav = h('nav', {});
    items.forEach(item => {
      nav.appendChild(h('div', {
        class: 'nav-item', 'data-route': item.route, title: `${item.label} (${item.shortcut})`,
        onClick: () => Router.navigate(item.route)
      }, renderIcon(Icons[item.icon]), h('span', { class: 'nav-label' }, item.label)));
    });

    const footer = h('div', { class: 'sidebar-footer' },
      h('div', { id: 'sidebar-version' }, 'llvm-advisor'),
    );

    const pinBtn = h('div', {
      class: 'nav-item', title: 'Pin sidebar',
      onClick: () => State.set('sidebarPinned', !State.get('sidebarPinned'))
    }, renderIcon(Icons.pin), h('span', { class: 'nav-label' }, 'Pin'));

    const sb = h('aside', { id: 'sidebar' }, nav, pinBtn, footer);
    sb.addEventListener('mouseenter', () => { if (!State.get('sidebarPinned')) sb.parentElement.classList.add('sb-exp'); });
    sb.addEventListener('mouseleave', () => { if (!State.get('sidebarPinned')) sb.parentElement.classList.remove('sb-exp'); });
    return sb;
  },

  updateNav() {
    const route = State.get('route');
    document.querySelectorAll('.nav-item[data-route]').forEach(el => {
      el.classList.toggle('active', el.dataset.route === route);
    });
  },

  updateStatus(health) {
    const pill = document.getElementById('status-pill');
    if (!pill || !health) return;
    const dot = pill.querySelector('.status-dot');
    const text = pill.querySelector('span:last-child');
    dot.className = 'status-dot ' + (health.ok !== false ? 'ok' : 'err');
    text.textContent = health.ok !== false ? `serving · ${health.snapshots ?? 0} snaps` : 'error';
  },

  updateThemeIcon() {
    const icon = document.querySelector('#theme-toggle .icon');
    if (icon) icon.innerHTML = Icons[Theme.icon()];
  },

  updateSelectors() {
    const snapMenu = document.querySelector('#snapshot-dropdown .dd-menu');
    if (!snapMenu) return;
    const snaps = State.get('snapshots') || [];
    clearEl(snapMenu);
    if (snaps.length === 0) {
      snapMenu.appendChild(h('div', { class: 'dd-item text-muted' }, 'No snapshots'));
      return;
    }
    snaps.forEach(s => {
      snapMenu.appendChild(h('div', {
        class: 'dd-item' + (State.get('currentSnapshot')?.id === s.id ? ' selected' : ''),
        onClick: () => { State.set('currentSnapshot', s); this.closeDropdowns(); Router.resolve(); }
      }, `${(s.id || '').slice(0, 8)} · ${timeAgo(s.created_unix)}`));
    });
  },

  toggleDropdown(e) {
    const dd = e.target.closest('.dropdown');
    const wasOpen = dd.classList.contains('open');
    this.closeDropdowns();
    if (!wasOpen) dd.classList.add('open');
  },

  closeDropdowns() {
    document.querySelectorAll('.dropdown.open').forEach(d => d.classList.remove('open'));
  },

  renderMain(content) {
    const main = document.getElementById('main');
    if (!main) return;
    clearEl(main);
    if (typeof content === 'string') {
      const pre = document.createElement('pre');
      pre.textContent = content;
      main.appendChild(pre);
    } else if (content) {
      main.appendChild(content);
    }
  },

  renderDetail(content, title) {
    const panel = document.getElementById('detail-panel');
    if (!panel) return;
    clearEl(panel);
    if (!content) { State.set('detailOpen', false); return; }

    const card = h('div', { class: 'detail-card' });
    const header = h('div', { class: 'detail-card-header' },
      h('span', { class: 'detail-card-title' }, title || 'Details'),
      h('button', {
        class: 'detail-card-close',
        onClick: () => State.set('detailOpen', false),
        title: 'Close'
      }, '×')
    );
    const body = h('div', { class: 'detail-card-body' });
    body.appendChild(content);

    card.appendChild(header);
    card.appendChild(body);
    panel.appendChild(card);
    State.set('detailOpen', true);
  },
};

// Close dropdowns on outside click
document.addEventListener('click', e => {
  if (!e.target.closest('.dropdown')) Shell.closeDropdowns();
});

/* --- Command Palette --- */
const CommandPalette = {
  commands: [
    { label: 'Go to Overview', shortcut: 'g o', action: () => Router.navigate('/') },
    { label: 'Go to Units', shortcut: 'g u', action: () => Router.navigate('/units') },
    { label: 'Go to Compare', shortcut: 'g c', action: () => Router.navigate('/compare') },
    { label: 'Go to Timeline', shortcut: 'g t', action: () => Router.navigate('/timeline') },
    { label: 'Go to Insights', shortcut: 'g i', action: () => Router.navigate('/insights') },
    { label: 'Go to Settings', shortcut: 'g s', action: () => Router.navigate('/settings') },
  ],

  render() {
    const results = h('div', { class: 'cmd-results' });
    this.commands.forEach((cmd, i) => {
      results.appendChild(h('div', {
        class: 'cmd-item' + (i === 0 ? ' selected' : ''),
        onClick: () => { cmd.action(); State.set('commandPaletteOpen', false); }
      },
        h('span', { class: 'cmd-label' }, cmd.label),
        h('span', { class: 'cmd-shortcut' }, cmd.shortcut)
      ));
    });

    const input = h('input', { class: 'cmd-input', placeholder: 'Type a command…' });
    input.addEventListener('input', () => {
      const q = input.value.trim();
      const items = Array.from(results.querySelectorAll('.cmd-item'));
      items.forEach(el => {
        const text = el.querySelector('.cmd-label')?.textContent || el.textContent;
        const match = fuzzyMatch(q, text);
        el.style.display = match ? '' : 'none';
        el.dataset.score = match ? match.score : -999;
      });
      items.sort((a, b) => Number(b.dataset.score) - Number(a.dataset.score));
      items.forEach(el => results.appendChild(el));
    });
    input.addEventListener('keydown', e => {
      if (e.key === 'Escape') State.set('commandPaletteOpen', false);
      if (e.key === 'Enter') {
        const sel = results.querySelector('.cmd-item:not([style*="none"])');
        if (sel) sel.click();
      }
    });

    const overlay = h('div', { class: 'cmd-overlay', onClick: e => {
      if (e.target === overlay) State.set('commandPaletteOpen', false);
    }}, h('div', { class: 'cmd-box' }, input, results));
    return overlay;
  },
};
